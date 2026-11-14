# Stage 2b: frequency-model extensions on top of the v1 GLM of R/fit_glm.R.
#
#   1. GBM-informed refinement: bonus-malus "off claim-free ladder" flag,
#      spline alternatives to the bands, interactions tested by LRT.
#   2. Distribution choice on the final mean structure: Poisson vs
#      quasi-Poisson vs negative binomial (MASS::glm.nb).
#   3. Out-of-fold predictions of the v1 and refined GLMs for 5-fold CV.
#
# Usage (from the project root):  Rscript R/fit_freq_extensions.R
# Inputs : data/processed/train.csv, test.csv,
#          reports/tables/freq_term_tests_final_model.csv (v1 selected terms)
# Outputs: reports/tables/freq_refinement_steps.csv, freq_distribution_comparison.csv,
#          freq_refined_*.csv, reports/figures/glm_refined_age_bm.png,
#          data/processed/glm_refined_test_predictions.csv, glm_cv_predictions.csv
#
# Model selection uses training-data criteria only (likelihood-ratio tests,
# BIC). Test-set columns are reported for information, not used to choose.
#
# Role of the splines: they are a *diagnostic* of the bands, not candidates
# for the tariff. If a smooth curve in place of the bands gains little, the
# bands are adequate and the tariff stays a table that can be filed and read.

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(MASS)
  library(splines)
})
source(file.path("R", "glm_functions.R"))

t0 <- Sys.time()
tab_dir <- file.path("reports", "tables")
fig_dir <- file.path("reports", "figures")
factors <- c("DrivAgeBand", "VehAgeBand", "BonusMalusBand", "VehPowerBand", "VehBrand", "VehGas", "Region")
num_cols <- c("IDpol", "LogDensity", "DrivAge", "BonusMalus", "BMOffLadder", "ClaimNb", "Exposure")

read_set <- function(path, extra = character(0)) {
  as.data.frame(fread(path, select = c(factors, num_cols, extra), colClasses = list(character = factors)))
}
train <- read_set(file.path("data", "processed", "train.csv"), extra = "CVFold")
test <- read_set(file.path("data", "processed", "test.csv"))
logdens_ref <- weighted_median(train$LogDensity, train$Exposure)
for (nm in c("train", "test")) {
  d <- get(nm)
  d$LogDensity_c <- d$LogDensity - logdens_ref
  d$BM100plus <- as.integer(d$BonusMalus >= 100)  # entry level (novices) or malus
  assign(nm, d)
}
train <- set_base_levels(train, factors, train$Exposure)
test <- align_levels(test, train, factors)
y_tr <- train$ClaimNb
y_te <- test$ClaimNb

# ---------------------------------------------------------------------------
# 1. Refinement steps (Poisson, fitted on rating cells)
# ---------------------------------------------------------------------------
v1_terms <- read.csv(file.path(tab_dir, "freq_term_tests_final_model.csv"))$term
mk_formula <- function(terms) reformulate(c(terms, "offset(log(Exposure))"), response = "ClaimNb")
age_spline <- "ns(DrivAge, knots = c(20, 23, 26, 30, 40, 50, 60, 70), Boundary.knots = c(18, 90))"
bm_spline <- "ns(BonusMalus, knots = c(60, 70, 80, 90, 100, 115), Boundary.knots = c(50, 150))"

#' Policy-level summary of one candidate model (train fit criteria + test metrics).
summarise_model <- function(name, m, parent_name = NA, parent = NULL, nested = TRUE) {
  mu_tr <- predict(m, newdata = train, type = "response")
  mu_te <- predict(m, newdata = test, type = "response")
  p <- sum(!is.na(coef(m)))
  ll <- poisson_loglik(y_tr, mu_tr)
  phi <- sum((y_tr - mu_tr)^2 / mu_tr) / (length(y_tr) - p)
  row <- data.frame(model = name, compared_with = parent_name, parameters = p, rating_cells = m$n_cells,
                    train_loglik = ll, aic = -2 * ll + 2 * p, bic = -2 * ll + log(length(y_tr)) * p,
                    pearson_dispersion = phi, lr = NA_real_, lr_df = NA_real_, p_value = NA_real_,
                    p_value_quasi = NA_real_,
                    test_mean_poisson_deviance = poisson_deviance(y_te, mu_te) / length(y_te),
                    test_gini = gini_index(y_te, mu_te / test$Exposure, test$Exposure))
  if (!is.null(parent) && nested) {
    lr <- lr_compare(parent$train_loglik, ll, p - parent$parameters, phi = phi)
    row[c("lr", "lr_df", "p_value", "p_value_quasi")] <- as.list(lr)
  }
  row
}

steps <- list()
fits <- list()
fit_step <- function(key, name, terms, parent_key = NULL, nested = TRUE, start = NULL) {
  t1 <- Sys.time()
  m <- fit_poisson_cells(mk_formula(terms), train, start = start, epsilon = 1e-7)
  parent <- if (is.null(parent_key)) NULL else steps[[parent_key]]
  pname <- if (is.null(parent_key)) NA else steps[[parent_key]]$model
  steps[[key]] <<- summarise_model(name, m, pname, parent, nested)
  fits[[key]] <<- m
  cat(sprintf("  %-55s %5.1f s  BIC %.1f  test Gini %.4f\n", name,
              as.numeric(difftime(Sys.time(), t1, units = "secs")), steps[[key]]$bic, steps[[key]]$test_gini))
  invisible(m)
}

cat("Refinement steps:\n")
fit_step("v1", "v1: bands (R/fit_glm.R selection)", v1_terms)
m1_terms <- c(v1_terms, "BMOffLadder")
fit_step("m1", "+ BM off-ladder flag", m1_terms, "v1", start = coef(fits$v1))
fit_step("s_age", "+ flag, DrivAge natural spline instead of bands",
         c(setdiff(m1_terms, "DrivAgeBand"), age_spline), "m1", nested = FALSE, start = coef(fits$m1))
fit_step("s_bm", "+ flag, BonusMalus natural spline instead of bands",
         c(setdiff(m1_terms, "BonusMalusBand"), bm_spline), "m1", nested = FALSE, start = coef(fits$m1))
fits$s_age <- NULL
fits$s_bm <- NULL
invisible(gc())

interactions <- list(
  # "BM100plus" as a main effect is aliased with the BM bands (it gets an NA
  # coefficient); it is written out so that the interaction is coded with the
  # usual treatment contrasts, relative to the base age band.
  age_bm = c("DrivAgeBand:BonusMalusBand", "BM100plus + DrivAgeBand:BM100plus"),
  vehage_gas = c("VehAgeBand:VehGas")
)
# The compact age x (BM >= 100) candidate is kept to show its effect in the figure.
show_ia <- "BM100plus + DrivAgeBand:BM100plus"
for (grp in names(interactions)) {
  for (ia in interactions[[grp]]) {
    fit_step(ia, paste("+ flag +", sub("BM100plus + ", "", ia, fixed = TRUE)), c(m1_terms, ia), "m1",
             start = coef(fits$m1))
    if (ia != show_ia) fits[[ia]] <- NULL
    invisible(gc())
  }
}
ia_fit <- fits[[show_ia]]
fits[[show_ia]] <- NULL

# Selection rule (training data only): within each group keep the candidate
# with the lowest BIC, provided it lowers BIC versus the flag model and its
# quasi-Poisson LR p-value is below 1%.
chosen <- character(0)
for (grp in names(interactions)) {
  cand <- do.call(rbind, steps[interactions[[grp]]])
  cand$term <- interactions[[grp]]
  best <- cand[which.min(cand$bic), ]
  if (best$bic < steps$m1$bic && best$p_value_quasi < 0.01) chosen <- c(chosen, best$term)
}
final_terms <- c(m1_terms, chosen)
cat("Chosen interactions:", if (length(chosen)) paste(chosen, collapse = ", ") else "none", "\n")

# Final refined model: full (non-slim) fit so that standard errors and term tests are available.
freq_ref <- fit_poisson_cells(mk_formula(final_terms), train, start = coef(fits$m1), keep_qr = TRUE)
fits$m1 <- NULL
v1_fit <- fits$v1
invisible(gc())
ref_row <- summarise_model("refined: + flag + chosen interactions", freq_ref, steps$v1$model, steps$v1)
steps$final <- ref_row
step_tab <- do.call(rbind, steps)
rownames(step_tab) <- NULL
write.csv(step_tab, file.path(tab_dir, "freq_refinement_steps.csv"), row.names = FALSE)

mu_ref_tr <- predict(freq_ref, newdata = train, type = "response")
phi_ref <- ref_row$pearson_dispersion
rel_ref <- relativity_table(freq_ref)
rel_ref$se_quasi <- rel_ref$se * sqrt(phi_ref)
rel_ref$lower95_quasi <- exp(rel_ref$beta - 1.96 * rel_ref$se_quasi)
rel_ref$upper95_quasi <- exp(rel_ref$beta + 1.96 * rel_ref$se_quasi)
write.csv(rel_ref, file.path(tab_dir, "freq_refined_relativities.csv"), row.names = FALSE)

# ---------------------------------------------------------------------------
# 2. Distribution: Poisson vs quasi-Poisson vs negative binomial (final terms)
# ---------------------------------------------------------------------------
# NB2 likelihood is not a function of cell totals, so glm.nb is fitted at
# policy level. Warm start: Poisson coefficients and the ML theta given the
# Poisson means.
t1 <- Sys.time()
theta0 <- as.numeric(theta.ml(y_tr, mu_ref_tr, limit = 50))
nb_data <- train[, c(all.vars(mk_formula(final_terms)))]
freq_nb <- glm.nb(mk_formula(final_terms), data = nb_data, start = ifelse(is.na(coef(freq_ref)), 0, coef(freq_ref)), init.theta = theta0,
                  model = FALSE, control = glm.control(epsilon = 1e-6, maxit = 50))
nb_seconds <- as.numeric(difftime(Sys.time(), t1, units = "secs"))
rm(nb_data)
mu_nb_tr <- fitted(freq_nb)
freq_nb <- slim_glm(freq_nb, keep_qr = FALSE)
invisible(gc())
mu_ref_te <- predict(freq_ref, newdata = test, type = "response")
mu_nb_te <- predict(freq_nb, newdata = test, type = "response")
theta <- freq_nb$theta
ll_pois <- poisson_loglik(y_tr, mu_ref_tr)
ll_nb <- nb_loglik(y_tr, mu_nb_tr, theta)
p_ref <- ref_row$parameters
lr_nb <- 2 * (ll_nb - ll_pois)
rel_diff <- abs(exp(coef(freq_nb)[-1] - coef(freq_ref)[-1]) - 1)
od_ref <- overdispersion_test(y_tr, mu_ref_tr)
dist_tab <- data.frame(
  model = c("Poisson", "quasi-Poisson", "negative binomial (NB2, MASS::glm.nb)"),
  parameters = c(p_ref, p_ref + 1, p_ref + 1),
  dispersion_or_theta = c(1, phi_ref, theta),
  theta_se = c(NA, NA, freq_nb$SE.theta),
  train_loglik = c(ll_pois, NA, ll_nb),
  aic = c(-2 * ll_pois + 2 * p_ref, NA, -2 * ll_nb + 2 * (p_ref + 1)),
  train_fitted_over_observed = c(sum(mu_ref_tr), sum(mu_ref_tr), sum(mu_nb_tr)) / sum(y_tr),
  test_mean_poisson_deviance = c(rep(poisson_deviance(y_te, mu_ref_te), 2), poisson_deviance(y_te, mu_nb_te)) / length(y_te),
  test_mean_loglik = c(poisson_loglik(y_te, mu_ref_te), NA, nb_loglik(y_te, mu_nb_te, theta)) / length(y_te),
  test_gini = c(rep(gini_index(y_te, mu_ref_te / test$Exposure, test$Exposure), 2),
                gini_index(y_te, mu_nb_te / test$Exposure, test$Exposure)),
  se_multiplier_vs_poisson = c(1, sqrt(phi_ref), NA)
)
write.csv(dist_tab, file.path(tab_dir, "freq_distribution_comparison.csv"), row.names = FALSE)
nb_extra <- data.frame(
  metric = c("lr_nb_vs_poisson", "p_value_boundary_chibar", "max_abs_relativity_change_nb_vs_poisson",
             "median_abs_relativity_change_nb_vs_poisson", "nb_fit_seconds", "theta_start_poisson_means",
             "cameron_trivedi_alpha_refined", "cameron_trivedi_p_refined",
             "nb_implied_variance_ratio_at_mean_policy"),
  value = c(lr_nb, 0.5 * pchisq(lr_nb, 1, lower.tail = FALSE), max(rel_diff), median(rel_diff), nb_seconds,
            theta0, od_ref$alpha, od_ref$p_value, 1 + mean(mu_nb_tr) / theta)
)
write.csv(nb_extra, file.path(tab_dir, "freq_nb_details.csv"), row.names = FALSE)
cat(sprintf("glm.nb: theta = %.3f (SE %.3f), LR vs Poisson = %.1f, %.1f s\n", theta, freq_nb$SE.theta, lr_nb, nb_seconds))

fwrite(data.frame(IDpol = test$IDpol, freq_glm_refined = mu_ref_te / test$Exposure,
                  freq_glm_nb = mu_nb_te / test$Exposure),
       file.path("data", "processed", "glm_refined_test_predictions.csv"))
rm(freq_nb)
invisible(gc())

# ---------------------------------------------------------------------------
# 3. Refined tariff: age effect by bonus-malus group, and build-up examples
# ---------------------------------------------------------------------------
age_levels <- c("18-20", "21-22", "23-25", "26-30", "31-40", "41-50", "51-60", "61-70", "71-90")
grid <- expand.grid(DrivAgeBand = age_levels, BM = c(50, 100), stringsAsFactors = FALSE)
base_row <- train[1, ]
for (v in factors) base_row[[v]] <- factor(levels(train[[v]])[1], levels = levels(train[[v]]))
nd <- base_row[rep(1, nrow(grid)), ]
nd$DrivAgeBand <- factor(grid$DrivAgeBand, levels = levels(train$DrivAgeBand))
nd$BonusMalusBand <- factor(ifelse(grid$BM == 50, "50", "100"), levels = levels(train$BonusMalusBand))
nd$BonusMalus <- grid$BM
nd$BM100plus <- as.integer(grid$BM >= 100)
nd$BMOffLadder <- 0L
nd$LogDensity_c <- 0
nd$Exposure <- 1
ia_label <- "candidate + age x (BM >= 100): LRT significant, rejected by BIC"
age_tab <- rbind(
  data.frame(model = "v1", grid, frequency = predict(v1_fit, nd, type = "response")),
  data.frame(model = "refined (selected)", grid, frequency = predict(freq_ref, nd, type = "response")),
  data.frame(model = ia_label, grid, frequency = suppressWarnings(predict(ia_fit, nd, type = "response")))
)
rm(ia_fit)
age_tab$relativity_to_41_50 <- ave(seq_len(nrow(age_tab)), age_tab$model, age_tab$BM, FUN = function(i) {
  age_tab$frequency[i] / age_tab$frequency[i][age_tab$DrivAgeBand[i] == "41-50"]
})
write.csv(age_tab, file.path(tab_dir, "freq_refined_age_by_bm.csv"), row.names = FALSE)

age_tab$DrivAgeBand <- factor(age_tab$DrivAgeBand, levels = age_levels)
age_tab$group <- ifelse(age_tab$BM == 50, "BonusMalus 50 (claim-free maximum bonus)", "BonusMalus 100 (entry level)")
p_age <- ggplot(age_tab, aes(DrivAgeBand, frequency, colour = group, linetype = model, group = interaction(group, model))) +
  geom_line(linewidth = 0.8) + geom_point(size = 1.8) +
  scale_colour_manual(values = c("#2a78d6", "#eb6834"), name = NULL) +
  scale_linetype_manual(values = setNames(c("dotted", "solid", "dashed"), c("v1", "refined (selected)", ia_label)),
                        name = NULL) +
  scale_y_log10() +
  labs(x = "driver age band", y = "expected claims per year (log scale)",
       title = "Driver age effect by bonus-malus level",
       subtitle = paste("Base profile for all other factors. Without an interaction the two curves are parallel",
                        "on the log scale;\nthe candidate lets the young-driver loading differ between bonus and entry level.")) +
  theme_minimal(base_size = 10) +
  theme(legend.position = "bottom", legend.box = "vertical", panel.grid.minor = element_blank(),
        plot.title = element_text(face = "bold"), plot.background = element_rect(fill = "#fcfcfb", colour = NA))
ggsave(file.path(fig_dir, "glm_refined_age_bm.png"), p_age, width = 8, height = 5, dpi = 150)

profiles <- read.csv(file.path(tab_dir, "tariff_profiles.csv"), stringsAsFactors = FALSE)
profiles$BonusMalus <- as.numeric(profiles$BonusMalusBand)  # the example profiles sit on 50 / 100
profiles$BM100plus <- as.integer(profiles$BonusMalus >= 100)
profiles$BMOffLadder <- 0L
profiles$LogDensity_c <- log(profiles$Density) - logdens_ref
profiles$Exposure <- 1
profiles <- align_levels(profiles, train, factors)
bf <- relativity_buildup(freq_ref, profiles)
fp <- predict(freq_ref, profiles, type = "response")
stopifnot(isTRUE(all.equal(unname(apply(bf, 1, prod)), unname(fp))))
long <- rbind(data.frame(component = colnames(bf), t(bf), check.names = FALSE),
              data.frame(component = "= expected frequency (refined)", t(fp), check.names = FALSE))
colnames(long) <- c("component", profiles$profile)
write.csv(long, file.path(tab_dir, "freq_refined_tariff_examples.csv"), row.names = FALSE)

# ---------------------------------------------------------------------------
# 4. Out-of-fold predictions for 5-fold cross-validation (train set only)
# ---------------------------------------------------------------------------
v1_start <- coef(v1_fit)
ref_start <- ifelse(is.na(coef(freq_ref)), 0, coef(freq_ref))
rm(freq_ref, v1_fit)
fits <- NULL
invisible(gc())
cv_pred <- vector("list", max(train$CVFold))
for (k in sort(unique(train$CVFold))) {
  t1 <- Sys.time()
  fit_set <- train[train$CVFold != k, ]
  hold <- train[train$CVFold == k, ]
  m_v1 <- fit_poisson_cells(mk_formula(v1_terms), fit_set, start = v1_start, epsilon = 1e-7)
  m_ref <- fit_poisson_cells(mk_formula(final_terms), fit_set, start = ref_start, epsilon = 1e-7)
  cv_pred[[k]] <- data.frame(IDpol = hold$IDpol, CVFold = k,
                             freq_glm = predict(m_v1, hold, type = "response") / hold$Exposure,
                             freq_glm_refined = predict(m_ref, hold, type = "response") / hold$Exposure)
  rm(m_v1, m_ref, fit_set, hold)
  invisible(gc())
  cat(sprintf("  CV fold %d: %.1f s\n", k, as.numeric(difftime(Sys.time(), t1, units = "secs"))))
}
fwrite(do.call(rbind, cv_pred), file.path("data", "processed", "glm_cv_predictions.csv"))

writeLines(c(paste("Refined frequency formula:", deparse(mk_formula(final_terms), width.cutoff = 500)),
             sprintf("Extension stage runtime: %.1f s", as.numeric(difftime(Sys.time(), t0, units = "secs")))),
           file.path(tab_dir, "glm_extensions_log.txt"))
cat(sprintf("Extension stage runtime: %.1f s\n", as.numeric(difftime(Sys.time(), t0, units = "secs"))))
