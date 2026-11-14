# Stage 2: frequency (Poisson) and severity (Gamma) GLMs, relativities,
# variable selection, overdispersion check, test-set predictions and tariff.
#
# Usage (from the project root):  Rscript R/fit_glm.R
# Inputs : data/processed/train.csv, test.csv, reports/tables/large_loss.csv
# Outputs: reports/tables/*.csv|txt, reports/figures/glm_relativities.png,
#          data/processed/glm_test_predictions.csv

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
})
source(file.path("R", "glm_functions.R"))

t0 <- Sys.time()
tab_dir <- file.path("reports", "tables")
fig_dir <- file.path("reports", "figures")
band_cols <- c("DrivAgeBand", "VehAgeBand", "BonusMalusBand", "VehPowerBand")
factors <- c(band_cols, "VehBrand", "VehGas", "Region", "Area")

read_set <- function(path) {
  as.data.frame(fread(path, colClasses = list(character = c(band_cols, "VehBrand", "VehGas", "Region", "Area"))))
}
train <- read_set(file.path("data", "processed", "train.csv"))
test <- read_set(file.path("data", "processed", "test.csv"))
large_loss <- read.csv(file.path(tab_dir, "large_loss.csv"))
ll_loading <- large_loss$large_loss_loading

# Centre log-density at the exposure-weighted median so that the base profile
# has a typical density instead of 1 inhabitant/km2.
logdens_ref <- weighted_median(train$LogDensity, train$Exposure)
train$LogDensity_c <- train$LogDensity - logdens_ref
test$LogDensity_c <- test$LogDensity - logdens_ref
train <- set_base_levels(train, factors, train$Exposure)
test <- align_levels(test, train, factors)

log_lines <- c(sprintf("Log-density reference (exposure-weighted median): %.3f, i.e. Density = %.0f inhabitants/km2",
                       logdens_ref, exp(logdens_ref)))

# ---------------------------------------------------------------------------
# 1. Frequency GLM: Poisson, log link, offset log(Exposure)
# ---------------------------------------------------------------------------
# Rows with identical rating factors are aggregated (sum of claims, sum of
# exposure). For a Poisson GLM the claim totals per cell are sufficient
# statistics, so coefficients, standard errors and likelihood-ratio statistics
# are identical to the policy-level fit, at a fraction of the run time.
cell_vars <- c(factors, "LogDensity_c")
agg <- as.data.table(train)[, .(ClaimNb = sum(ClaimNb), Exposure = sum(Exposure)), by = cell_vars]
agg <- as.data.frame(agg)
for (v in factors) agg[[v]] <- factor(agg[[v]], levels = levels(train[[v]]))
log_lines <- c(log_lines, sprintf("Frequency GLM fitted on %d rating cells (from %d policies)", nrow(agg), nrow(train)))

freq_full <- glm(
  ClaimNb ~ DrivAgeBand + VehAgeBand + BonusMalusBand + VehPowerBand + VehBrand +
    VehGas + Region + Area + LogDensity_c + offset(log(Exposure)),
  family = poisson(link = "log"), data = agg
)
freq_null <- glm(ClaimNb ~ 1 + offset(log(Exposure)), family = poisson(), data = agg)

# Overdispersion on the full model, measured at policy level.
mu_full_policy <- predict(freq_full, newdata = train, type = "response")
p <- length(coef(freq_full))
phi_policy <- sum((train$ClaimNb - mu_full_policy)^2 / mu_full_policy) / (nrow(train) - p)
lrt_full <- term_tests(freq_full, test = "LRT", phi = 1)
lrt_full$p_value_quasi <- pchisq(lrt_full$deviance_increase / phi_policy, lrt_full$df, lower.tail = FALSE)
write.csv(lrt_full, file.path(tab_dir, "freq_term_tests_full_model.csv"), row.names = FALSE)

sel <- backward_select(freq_full, test = "LRT", alpha = 0.05, initial_tests = lrt_full)
freq_glm <- sel$model
write.csv(sel$steps, file.path(tab_dir, "freq_backward_steps.csv"), row.names = FALSE)
log_lines <- c(log_lines, "Frequency backward elimination (LRT, alpha = 0.05):",
               if (nrow(sel$steps)) capture.output(print(sel$steps, row.names = FALSE)) else "  no term dropped",
               paste("Final frequency formula:", deparse(formula(freq_glm), width.cutoff = 500)))

# Policy-level diagnostics of the selected model
mu_train <- predict(freq_glm, newdata = train, type = "response")
phi_final <- sum((train$ClaimNb - mu_train)^2 / mu_train) / (nrow(train) - length(coef(freq_glm)))
lrt_final <- term_tests(freq_glm, test = "LRT", phi = 1)
lrt_final$p_value_quasi <- pchisq(lrt_final$deviance_increase / phi_final, lrt_final$df, lower.tail = FALSE)
write.csv(lrt_final, file.path(tab_dir, "freq_term_tests_final_model.csv"), row.names = FALSE)
od <- overdispersion_test(train$ClaimNb, mu_train)
diag_freq <- data.frame(
  metric = c("policies", "parameters", "pearson_dispersion", "deviance_dispersion",
             "cameron_trivedi_alpha", "cameron_trivedi_t", "cameron_trivedi_p",
             "train_sum_observed_claims", "train_sum_fitted_claims",
             "train_poisson_deviance_null", "train_poisson_deviance_model", "aic_cells_model"),
  value = c(nrow(train), length(coef(freq_glm)),
            phi_final,
            poisson_deviance(train$ClaimNb, mu_train) / (nrow(train) - length(coef(freq_glm))),
            od$alpha, od$t, od$p_value,
            sum(train$ClaimNb), sum(mu_train),
            poisson_deviance(train$ClaimNb, rep(sum(train$ClaimNb) / sum(train$Exposure), nrow(train)) * train$Exposure),
            poisson_deviance(train$ClaimNb, mu_train), AIC(freq_glm))
)
write.csv(diag_freq, file.path(tab_dir, "freq_glm_diagnostics.csv"), row.names = FALSE)

rel_freq <- relativity_table(freq_glm)
write.csv(rel_freq, file.path(tab_dir, "freq_glm_relativities.csv"), row.names = FALSE)
log_lines <- c(log_lines, sprintf("Frequency base (exp intercept) = %.5f claims per year", exp(coef(freq_glm)[1])))

# ---------------------------------------------------------------------------
# 2. Severity GLM: Gamma, log link, average capped claim, weights = claim count
# ---------------------------------------------------------------------------
sev_train <- train[train$NbClaimsSev > 0, ]
sev_train$AvgClaimCapped <- sev_train$ClaimAmountCappedTotal / sev_train$NbClaimsSev
sev_full <- glm(
  AvgClaimCapped ~ DrivAgeBand + VehAgeBand + BonusMalusBand + VehPowerBand + VehBrand +
    VehGas + Region + Area + LogDensity_c,
  family = Gamma(link = "log"), weights = NbClaimsSev, data = sev_train,
  control = glm.control(maxit = 100)
)
phi_sev_full <- pearson_dispersion(sev_full)
ftest_full <- term_tests(sev_full, test = "F", phi = phi_sev_full)
write.csv(ftest_full, file.path(tab_dir, "sev_term_tests_full_model.csv"), row.names = FALSE)

# Gamma: dispersion is estimated, so terms are tested with F statistics; the
# Pearson dispersion is re-estimated on the current model at every step.
sev_sel <- backward_select(sev_full, test = "F", alpha = 0.05, initial_tests = ftest_full)
sev_glm <- sev_sel$model
sev_steps <- sev_sel$steps
write.csv(sev_steps, file.path(tab_dir, "sev_backward_steps.csv"), row.names = FALSE)
rel_sev <- relativity_table(sev_glm)
write.csv(rel_sev, file.path(tab_dir, "sev_glm_relativities.csv"), row.names = FALSE)
sev_fitted <- fitted(sev_glm)
diag_sev <- data.frame(
  metric = c("policies_with_claims", "claims", "parameters", "pearson_dispersion",
             "train_weighted_mean_observed", "train_weighted_mean_fitted"),
  value = c(nrow(sev_train), sum(sev_train$NbClaimsSev), length(coef(sev_glm)), pearson_dispersion(sev_glm),
            weighted.mean(sev_train$AvgClaimCapped, sev_train$NbClaimsSev),
            weighted.mean(sev_fitted, sev_train$NbClaimsSev))
)
write.csv(diag_sev, file.path(tab_dir, "sev_glm_diagnostics.csv"), row.names = FALSE)
log_lines <- c(log_lines, "Severity backward elimination (F test, alpha = 0.05):",
               capture.output(print(sev_steps, row.names = FALSE)),
               paste("Final severity formula:", deparse(formula(sev_glm), width.cutoff = 500)),
               sprintf("Severity base (exp intercept) = %.2f EUR", exp(coef(sev_glm)[1])))

# ---------------------------------------------------------------------------
# 2b. Tweedie GLM on the pure premium (comparison only)
# ---------------------------------------------------------------------------
# Compound Poisson-Gamma model of capped claim cost per exposure year, weights
# = exposure, log link, family from mgcv (a recommended package shipped with
# R). Power p = 1.5 is fixed, not estimated. Terms = union of those selected
# for frequency and severity.
tw_terms <- union(attr(terms(freq_glm), "term.labels"), attr(terms(sev_glm), "term.labels"))
train$PurePremiumCapped <- train$ClaimAmountCappedTotal / train$Exposure
tweedie_glm <- glm(reformulate(tw_terms, response = "PurePremiumCapped"),
                   family = mgcv::Tweedie(p = 1.5, link = "log"), weights = Exposure, data = train,
                   control = glm.control(maxit = 100))
log_lines <- c(log_lines, sprintf("Tweedie GLM (p = 1.5): converged = %s in %d iterations; train fitted / observed capped cost = %.4f",
                                  tweedie_glm$converged, tweedie_glm$iter,
                                  sum(fitted(tweedie_glm) * train$Exposure) / sum(train$ClaimAmountCappedTotal)))

# ---------------------------------------------------------------------------
# 3. Test-set predictions (annual frequency, capped severity, pure premium)
# ---------------------------------------------------------------------------
test1 <- test
test1$Exposure <- 1
pred <- data.frame(
  IDpol = test$IDpol,
  freq_glm = predict(freq_glm, newdata = test1, type = "response"),
  sev_glm = predict(sev_glm, newdata = test, type = "response")
)
pred$pure_premium_glm <- pred$freq_glm * pred$sev_glm * ll_loading
pred$pure_premium_tweedie <- predict(tweedie_glm, newdata = test, type = "response") * ll_loading
fwrite(pred, file.path("data", "processed", "glm_test_predictions.csv"))

# ---------------------------------------------------------------------------
# 4. Tariff example: multiplicative build-up for three profiles
# ---------------------------------------------------------------------------
profiles <- data.frame(
  profile = c("A: experienced, urban", "B: young new driver", "C: senior, rural"),
  DrivAgeBand = c("41-50", "18-20", "71-90"),
  VehAgeBand = c("4-11", "12-15", "0-3"),
  BonusMalusBand = c("50", "100", "50"),
  VehPowerBand = c("5-6", "5-6", "7-8"),
  VehBrand = c("B1", "B12", "B2"),
  VehGas = c("Diesel", "Regular", "Regular"),
  Region = c("Ile-de-France", "Rhone-Alpes", "Bretagne"),
  Area = c("E", "D", "B"),
  Density = c(3000, 800, 60),
  Exposure = 1
)
profiles$LogDensity_c <- log(profiles$Density) - logdens_ref
profiles <- align_levels(profiles, train, factors)
bf <- relativity_buildup(freq_glm, profiles)
bs <- relativity_buildup(sev_glm, profiles)
freq_p <- predict(freq_glm, newdata = profiles, type = "response")
sev_p <- predict(sev_glm, newdata = profiles, type = "response")
stopifnot(isTRUE(all.equal(unname(apply(bf, 1, prod)), unname(freq_p))),
          isTRUE(all.equal(unname(apply(bs, 1, prod)), unname(sev_p))))
long <- rbind(
  data.frame(component = paste0("frequency: ", colnames(bf)), t(bf), check.names = FALSE),
  data.frame(component = "= expected frequency", t(freq_p), check.names = FALSE),
  data.frame(component = paste0("severity: ", colnames(bs)), t(bs), check.names = FALSE),
  data.frame(component = "= expected capped severity (EUR)", t(sev_p), check.names = FALSE),
  data.frame(component = "x large-loss loading", t(rep(ll_loading, 3)), check.names = FALSE),
  data.frame(component = "= pure premium (EUR / year)", t(freq_p * sev_p * ll_loading), check.names = FALSE)
)
colnames(long) <- c("component", profiles$profile)
rownames(long) <- NULL
write.csv(profiles[, c("profile", factors, "Density")], file.path(tab_dir, "tariff_profiles.csv"), row.names = FALSE)
write.csv(long, file.path(tab_dir, "tariff_examples.csv"), row.names = FALSE)

# ---------------------------------------------------------------------------
# 5. Relativity figure (frequency GLM)
# ---------------------------------------------------------------------------
band_order <- list(
  DrivAgeBand = c("18-20", "21-22", "23-25", "26-30", "31-40", "41-50", "51-60", "61-70", "71-90"),
  VehAgeBand = c("0-3", "4-11", "12-15", "16-20"),
  BonusMalusBand = c("50", "51-59", "60-69", "70-89", "90-99", "100", "101-150"),
  VehPowerBand = c("4", "5-6", "7-8", "9-15")
)
plot_terms <- intersect(c("DrivAgeBand", "BonusMalusBand", "VehAgeBand", "VehPowerBand", "VehBrand", "VehGas"),
                        rel_freq$term)
pd <- rel_freq[rel_freq$term %in% plot_terms, ]
pd$level <- as.character(pd$level)
pd$order <- mapply(function(t, l) {
  if (t %in% names(band_order)) match(l, band_order[[t]])
  else if (t == "VehBrand") as.integer(sub("B", "", l)) else match(l, c("Diesel", "Regular"))
}, pd$term, pd$level)
pd$level_f <- factor(paste(pd$term, pd$level), levels = unique(paste(pd$term, pd$level)[order(pd$term, pd$order)]))
pd$is_base <- is.na(pd$se)
pd$term <- factor(pd$term, levels = plot_terms)
lbl <- setNames(pd$level, paste(pd$term, pd$level))
p_rel <- ggplot(pd, aes(x = level_f, y = relativity)) +
  geom_hline(yintercept = 1, colour = "#a3a29c", linewidth = 0.4, linetype = "dashed") +
  geom_errorbar(aes(ymin = lower95, ymax = upper95), width = 0.25, colour = "#2a78d6", na.rm = TRUE) +
  geom_point(aes(shape = is_base), size = 2.2, colour = "#2a78d6") +
  scale_shape_manual(values = c(`FALSE` = 16, `TRUE` = 1), labels = c("estimate (95% CI)", "base level"), name = NULL) +
  scale_x_discrete(labels = lbl) +
  facet_wrap(~term, scales = "free", nrow = 2) +
  labs(x = NULL, y = "relativity exp(beta)",
       title = "Frequency GLM relativities (Poisson, log link, offset log exposure)",
       subtitle = sprintf("Base frequency %.4f claims/yr; log-density relativity %.3f per +1 (centred at Density %.0f)",
                          exp(coef(freq_glm)[1]), rel_freq$relativity[rel_freq$term == "LogDensity_c"], exp(logdens_ref))) +
  theme_minimal(base_size = 10) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "bottom",
        panel.grid.minor = element_blank(), plot.title = element_text(face = "bold"),
        plot.background = element_rect(fill = "#fcfcfb", colour = NA))
ggsave(file.path(fig_dir, "glm_relativities.png"), p_rel, width = 11, height = 6.5, dpi = 150)

log_lines <- c(log_lines, sprintf("R stage runtime: %.1f s", as.numeric(difftime(Sys.time(), t0, units = "secs"))))
writeLines(log_lines, file.path(tab_dir, "glm_log.txt"))
cat(log_lines, sep = "\n")
