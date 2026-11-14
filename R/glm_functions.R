# Helper functions for the frequency / severity GLMs (base R + data.table only).
# Sourced by R/fit_glm.R and by the testthat suite.

#' Set the reference level of each factor to its highest-exposure level.
#'
#' @param df data.frame with the factor columns and a weight column.
#' @param vars character vector of column names to convert to factors.
#' @param weight numeric vector (exposure or claim count) used to pick the base.
#' @return df with the columns converted to factors, base level first.
set_base_levels <- function(df, vars, weight) {
  for (v in vars) {
    x <- as.character(df[[v]])
    w <- tapply(weight, x, sum)
    base <- names(w)[which.max(w)]
    df[[v]] <- relevel(factor(x), ref = base)
  }
  df
}

#' Apply the training factor levels (same base) to a new data set.
align_levels <- function(newdf, ref_df, vars) {
  for (v in vars) newdf[[v]] <- factor(as.character(newdf[[v]]), levels = levels(ref_df[[v]]))
  newdf
}

#' Total Poisson deviance 2 * sum(y log(y / mu) - (y - mu)), with 0 log 0 = 0.
poisson_deviance <- function(y, mu) {
  stopifnot(length(y) == length(mu), all(mu > 0))
  term <- ifelse(y > 0, y * log(y / mu), 0)
  2 * sum(term - (y - mu))
}

#' Pearson dispersion statistic: sum of squared Pearson residuals / residual df.
pearson_dispersion <- function(model) {
  sum(residuals(model, type = "pearson")^2) / df.residual(model)
}

#' Cameron & Trivedi (1990) regression test for overdispersion in a Poisson model.
#'
#' Under H0 Var(Y) = mu; under H1 Var(Y) = mu + alpha * mu^2. The auxiliary
#' OLS regression ((y - mu)^2 - y) / mu = alpha * mu + error (no intercept)
#' gives a t-statistic for alpha; a one-sided p-value is returned.
overdispersion_test <- function(y, mu) {
  z <- ((y - mu)^2 - y) / mu
  aux <- lm(z ~ 0 + mu)
  est <- coef(summary(aux))
  t_stat <- est[1, "t value"]
  list(alpha = unname(est[1, "Estimate"]), t = unname(t_stat),
       p_value = unname(pnorm(t_stat, lower.tail = FALSE)))
}

#' Test every model term by dropping it (likelihood-ratio or F test).
#'
#' Each reduced model is refitted with glm.fit() from the full-model
#' coefficients (warm start), which is several times faster than drop1() on
#' large data and gives the same deviances.
#'
#' @param model a fitted glm.
#' @param test "LRT" (chi-square on the deviance increase, scaled by `phi`) or
#'   "F" (F = deviance increase / df / phi on (df, residual df)).
#' @param phi dispersion used to scale the statistic: 1 for a Poisson LRT, a
#'   Pearson estimate for a quasi-Poisson or Gamma model.
#' @return data.frame(term, df, deviance_increase, statistic, p_value)
term_tests <- function(model, test = c("LRT", "F"), phi = 1) {
  test <- match.arg(test)
  X <- model.matrix(model)
  a <- attr(X, "assign")
  b <- coef(model)
  b[is.na(b)] <- 0
  off <- model$offset
  labels <- attr(terms(model), "term.labels")
  rows <- lapply(seq_along(labels), function(j) {
    keep <- a != j
    fit <- glm.fit(X[, keep, drop = FALSE], model$y, weights = model$prior.weights,
                   offset = off, family = model$family, start = b[keep])
    dd <- fit$deviance - model$deviance
    df <- sum(!is.na(coef(model)[a == j]))
    if (test == "LRT") {
      stat <- dd / phi
      p <- pchisq(stat, df, lower.tail = FALSE)
    } else {
      stat <- dd / df / phi
      p <- pf(stat, df, df.residual(model), lower.tail = FALSE)
    }
    data.frame(term = labels[j], df = df, deviance_increase = dd, statistic = stat, p_value = p)
  })
  do.call(rbind, rows)
}

#' Refit a glm without one term, using the data stored in the model object.
drop_term <- function(model, term) {
  cl <- update(model, as.formula(paste(". ~ . -", term)), evaluate = FALSE)
  cl$data <- quote(.model_data)
  eval(cl, list2env(list(.model_data = model$data), parent = globalenv()))
}

#' Backward elimination: drop the least significant term while its p-value > alpha.
#'
#' @param phi dispersion for the tests. NULL means 1 for "LRT" and the Pearson
#'   estimate of the current model (re-estimated at every step) for "F".
#' @param initial_tests optional term_tests() result for `model`, to avoid
#'   recomputing it when it was already produced for reporting.
#' @return list(model = final glm, steps = data.frame of removed terms)
backward_select <- function(model, test = "LRT", alpha = 0.05, phi = NULL, initial_tests = NULL) {
  steps <- data.frame(step = integer(0), dropped = character(0), p_value = numeric(0))
  tt <- initial_tests
  repeat {
    if (is.null(tt)) {
      phi_k <- if (!is.null(phi)) phi else if (test == "F") pearson_dispersion(model) else 1
      tt <- term_tests(model, test = test, phi = phi_k)
    }
    if (max(tt$p_value) <= alpha) break
    worst <- tt$term[which.max(tt$p_value)]
    steps <- rbind(steps, data.frame(step = nrow(steps) + 1, dropped = worst,
                                     p_value = max(tt$p_value)))
    model <- drop_term(model, worst)
    tt <- NULL
  }
  list(model = model, steps = steps)
}

#' Coefficients as multiplicative relativities exp(beta) with 95% Wald intervals.
#'
#' Base levels are added explicitly with relativity 1 so that the table can be
#' read as a tariff. Numeric terms give the relativity for +1 unit.
relativity_table <- function(model) {
  est <- coef(summary(model))
  out <- list()
  # Column -> term map from a small model matrix (columns depend only on xlevels).
  tt <- delete.response(terms(model))
  mm_small <- model.matrix(tt, model.frame(tt, head(model$data, 50), xlev = model$xlevels),
                           contrasts.arg = model$contrasts)
  assign <- attr(mm_small, "assign")
  labels <- attr(tt, "term.labels")
  for (term in labels) {
    if (!term %in% names(model$xlevels) && !term %in% rownames(est)) {
      # Interaction or multi-column (e.g. spline) term: one row per coefficient.
      nm <- intersect(colnames(mm_small)[assign == match(term, labels)], rownames(est))
      if (!length(nm)) next  # aliased term (e.g. a main effect implied by the bands)
      out[[term]] <- data.frame(term = term, level = nm, beta = est[nm, 1], se = est[nm, 2])
      next
    }
    if (term %in% names(model$xlevels)) {
      lv <- model$xlevels[[term]]
      rows <- data.frame(term = term, level = lv, beta = 0, se = NA_real_)
      for (i in seq_along(lv)[-1]) {
        nm <- paste0(term, lv[i])
        if (nm %in% rownames(est)) {
          rows$beta[i] <- est[nm, 1]
          rows$se[i] <- est[nm, 2]
        }
      }
    } else {
      rows <- data.frame(term = term, level = "per +1 unit", beta = est[term, 1], se = est[term, 2])
    }
    out[[term]] <- rows
  }
  tab <- do.call(rbind, out)
  rownames(tab) <- NULL
  tab$relativity <- exp(tab$beta)
  tab$lower95 <- exp(tab$beta - 1.96 * tab$se)
  tab$upper95 <- exp(tab$beta + 1.96 * tab$se)
  tab
}

#' Multiplicative build-up of the prediction for each row of `newdata`.
#'
#' Returns a matrix with one column per model term plus the base value
#' exp(intercept); the row products equal predict(model, newdata, "response")
#' when the offset is zero (Exposure = 1).
relativity_buildup <- function(model, newdata) {
  tt <- delete.response(terms(model))
  mf <- model.frame(tt, newdata, xlev = model$xlevels)
  mm <- model.matrix(tt, mf, contrasts.arg = model$contrasts)
  b <- coef(model)
  b[is.na(b)] <- 0
  assign <- attr(mm, "assign")
  labels <- attr(tt, "term.labels")
  out <- matrix(NA_real_, nrow(mm), length(labels) + 1,
                dimnames = list(NULL, c("base", labels)))
  out[, "base"] <- exp(b[1])
  for (j in seq_along(labels)) {
    cols <- which(assign == j)
    out[, labels[j]] <- exp(mm[, cols, drop = FALSE] %*% b[cols])
  }
  out
}

#' Exposure-weighted median, used to centre log-density.
weighted_median <- function(x, w) {
  o <- order(x)
  cw <- cumsum(w[o]) / sum(w)
  x[o][which(cw >= 0.5)[1]]
}

#' Fit a Poisson GLM on rating cells and return a memory-light glm object.
#'
#' Rows are aggregated over every variable used on the right-hand side of
#' `formula` (claims and exposure summed per cell; identical Poisson estimates,
#' see the testthat suite). Large components that are not needed for
#' prediction, coefficients or deviance (residuals, working weights, ...) are
#' removed, including the QR decomposition: the object can still predict()
#' and report coefficients and deviance, but not standard errors. With
#' `keep_qr = TRUE` the full glm is returned (for summary() and term_tests()).
#'
#' @param formula Poisson formula with `offset(log(<exposure>))`.
#' @param data policy-level data.frame.
#' @param start optional named vector of starting coefficients (warm start);
#'   names not in the new model are ignored, new ones start at 0.
#' @param epsilon IRLS convergence tolerance on the relative deviance change.
fit_poisson_cells <- function(formula, data, start = NULL, response = "ClaimNb",
                              exposure = "Exposure", keep_qr = FALSE, epsilon = 1e-8) {
  vars <- setdiff(all.vars(formula), c(response, exposure))
  dt <- data.table::as.data.table(data[, c(vars, response, exposure)])
  agg <- as.data.frame(dt[, lapply(.SD, sum), by = vars, .SDcols = c(response, exposure)])
  for (v in vars) if (is.factor(data[[v]])) agg[[v]] <- factor(agg[[v]], levels = levels(data[[v]]))
  b0 <- NULL
  if (!is.null(start)) {
    start[is.na(start)] <- 0
    cols <- colnames(model.matrix(formula, agg[seq_len(min(nrow(agg), 2000)), , drop = FALSE]))
    b0 <- setNames(rep(0, length(cols)), cols)
    common <- intersect(cols, names(start))
    b0[common] <- start[common]
    if (!"(Intercept)" %in% common) b0 <- NULL
  }
  m <- glm(formula, family = poisson(link = "log"), data = agg, start = b0, model = FALSE,
           control = glm.control(epsilon = epsilon, maxit = 50))
  m$n_cells <- nrow(agg)
  if (keep_qr) m else slim_glm(m)
}

#' Drop the n-length components of a glm that are not needed afterwards.
slim_glm <- function(m, keep_qr = FALSE) {
  for (nm in c("residuals", "fitted.values", "effects", "weights", "prior.weights",
               "linear.predictors", "y", "model")) m[[nm]] <- NULL
  # predict() needs the full QR of a rank-deficient fit (to flag non-estimable rows).
  if (!keep_qr && m$rank == length(m$coefficients)) {
    m$qr <- structure(list(pivot = m$qr$pivot, rank = m$qr$rank), class = "qr")
  }
  m
}

#' Policy-level Poisson log-likelihood of a fitted frequency model.
poisson_loglik <- function(y, mu) sum(dpois(y, mu, log = TRUE))

#' Policy-level negative binomial (NB2) log-likelihood, Var(Y) = mu + mu^2 / theta.
nb_loglik <- function(y, mu, theta) sum(dnbinom(y, size = theta, mu = mu, log = TRUE))

#' Gini index from the ordered Lorenz curve (same definition as src/metrics.py).
gini_index <- function(y, pred_freq, exposure) {
  o <- order(pred_freq, method = "radix")
  x <- c(0, cumsum(exposure[o]) / sum(exposure))
  c <- c(0, cumsum(y[o]) / sum(y))
  1 - 2 * sum(diff(x) * (head(c, -1) + tail(c, -1)) / 2)
}

#' Likelihood-ratio comparison of two nested models from their policy-level log-likelihoods.
#'
#' Returns the LR statistic, the added parameters, the Poisson p-value and a
#' quasi-Poisson p-value (statistic divided by the dispersion `phi`).
lr_compare <- function(ll_small, ll_big, df, phi = 1) {
  lr <- 2 * (ll_big - ll_small)
  c(lr = lr, df = df, p_value = pchisq(lr, df, lower.tail = FALSE),
    p_value_quasi = pchisq(lr / phi, df, lower.tail = FALSE))
}
