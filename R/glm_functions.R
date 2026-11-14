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
  for (term in attr(terms(model), "term.labels")) {
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
