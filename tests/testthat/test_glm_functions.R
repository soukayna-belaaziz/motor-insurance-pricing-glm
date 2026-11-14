# Unit tests for R/glm_functions.R. Run from the project root with
#   Rscript -e 'testthat::test_dir("tests/testthat")'
source(testthat::test_path("..", "..", "R", "glm_functions.R"))

simulate_portfolio <- function(n = 5000, seed = 1) {
  set.seed(seed)
  df <- data.frame(
    age = factor(sample(c("young", "mid", "old"), n, replace = TRUE, prob = c(0.2, 0.5, 0.3))),
    fuel = factor(sample(c("D", "R"), n, replace = TRUE)),
    x = rnorm(n),
    Exposure = runif(n, 0.1, 1)
  )
  eta <- log(0.08) + c(young = 0.7, mid = 0, old = -0.2)[as.character(df$age)] +
    0.15 * (df$fuel == "D") + 0.1 * df$x
  df$ClaimNb <- rpois(n, df$Exposure * exp(eta))
  df
}

test_that("poisson_deviance matches glm deviance and is zero for a perfect fit", {
  df <- simulate_portfolio()
  m <- glm(ClaimNb ~ age + fuel + x + offset(log(Exposure)), family = poisson(), data = df)
  expect_equal(poisson_deviance(df$ClaimNb, fitted(m)), deviance(m), tolerance = 1e-8)
  y <- c(0, 1, 3)
  expect_equal(poisson_deviance(y, pmax(y, 1e-12)), 0, tolerance = 1e-9)
  expect_error(poisson_deviance(c(1, 2), c(1, 0)))
})

test_that("Poisson GLM with log link and intercept is balanced on the training data", {
  df <- simulate_portfolio()
  m <- glm(ClaimNb ~ age + fuel + x + offset(log(Exposure)), family = poisson(), data = df)
  expect_equal(sum(fitted(m)), sum(df$ClaimNb), tolerance = 1e-8)
  # The score equations also balance every factor level.
  by_level <- tapply(fitted(m), df$age, sum) - tapply(df$ClaimNb, df$age, sum)
  expect_true(all(abs(by_level) < 1e-6))
})

test_that("aggregating identical rating cells gives the same Poisson coefficients", {
  df <- simulate_portfolio()
  df$x <- round(df$x)  # make cells repeat
  m_pol <- glm(ClaimNb ~ age + fuel + x + offset(log(Exposure)), family = poisson(), data = df)
  agg <- aggregate(cbind(ClaimNb, Exposure) ~ age + fuel + x, data = df, FUN = sum)
  m_agg <- glm(ClaimNb ~ age + fuel + x + offset(log(Exposure)), family = poisson(), data = agg)
  expect_equal(coef(m_agg), coef(m_pol), tolerance = 1e-7)
  expect_equal(sqrt(diag(vcov(m_agg))), sqrt(diag(vcov(m_pol))), tolerance = 1e-6)
})

test_that("set_base_levels picks the highest-exposure level as reference", {
  df <- data.frame(f = c("a", "b", "b", "c"), w = c(5, 1, 1, 1))
  out <- set_base_levels(df, "f", df$w)
  expect_equal(levels(out$f)[1], "a")
  out2 <- set_base_levels(df, "f", c(1, 3, 3, 1))
  expect_equal(levels(out2$f)[1], "b")
})

test_that("relativity_table returns exp(beta) with base levels at 1", {
  df <- simulate_portfolio()
  m <- glm(ClaimNb ~ age + fuel + x + offset(log(Exposure)), family = poisson(), data = df)
  rt <- relativity_table(m)
  expect_equal(nrow(rt), 3 + 2 + 1)
  expect_equal(rt$relativity[rt$term == "age" & rt$level == levels(df$age)[1]], 1)
  expect_equal(rt$relativity[rt$term == "x"], unname(exp(coef(m)["x"])))
  expect_true(all(rt$lower95 <= rt$relativity + 1e-12 | is.na(rt$lower95)))
})

test_that("relativity_buildup multiplies back to the GLM prediction", {
  df <- simulate_portfolio()
  m <- glm(ClaimNb ~ age + fuel + x + offset(log(Exposure)), family = poisson(), data = df)
  nd <- data.frame(age = factor(c("young", "old"), levels = levels(df$age)),
                   fuel = factor(c("D", "R"), levels = levels(df$fuel)), x = c(0.5, -1), Exposure = 1)
  b <- relativity_buildup(m, nd)
  expect_equal(unname(apply(b, 1, prod)), unname(predict(m, nd, type = "response")), tolerance = 1e-10)
})

test_that("term_tests reproduces drop1 likelihood-ratio statistics", {
  df <- simulate_portfolio()
  df$noise <- rnorm(nrow(df))
  m <- glm(ClaimNb ~ age + fuel + noise + offset(log(Exposure)), family = poisson(), data = df)
  tt <- term_tests(m, test = "LRT")
  d1 <- drop1(m, test = "LRT")
  expect_equal(tt$deviance_increase, unname(d1$LRT[-1]), tolerance = 1e-6)
  expect_equal(tt$p_value, unname(d1$`Pr(>Chi)`[-1]), tolerance = 1e-6)
})

test_that("backward_select drops a pure-noise variable and keeps real effects", {
  df <- simulate_portfolio(n = 20000, seed = 3)
  df$noise <- factor(sample(letters[1:4], nrow(df), replace = TRUE))
  m <- glm(ClaimNb ~ age + fuel + noise + offset(log(Exposure)), family = poisson(), data = df)
  sel <- backward_select(m, test = "LRT", alpha = 0.05)
  expect_equal(sel$steps$dropped, "noise")
  expect_setequal(attr(terms(sel$model), "term.labels"), c("age", "fuel"))
})

test_that("overdispersion test detects negative-binomial data but not Poisson data", {
  set.seed(7)
  n <- 20000
  mu <- rep(0.5, n)
  y_pois <- rpois(n, mu)
  y_nb <- rnbinom(n, mu = mu, size = 0.5)
  expect_gt(overdispersion_test(y_pois, mu)$p_value, 0.01)
  expect_lt(overdispersion_test(y_nb, mu)$p_value, 0.001)
})

test_that("weighted Gamma GLM reproduces the claim-weighted mean severity", {
  set.seed(11)
  n <- 3000
  df <- data.frame(g = factor(sample(c("a", "b"), n, replace = TRUE)), n_claims = sample(1:3, n, replace = TRUE))
  df$avg <- rgamma(n, shape = 2 * df$n_claims, rate = 2 * df$n_claims / ifelse(df$g == "a", 1000, 1500))
  m <- glm(avg ~ g, family = Gamma(link = "log"), weights = n_claims, data = df)
  expect_equal(exp(coef(m)[[1]]), weighted.mean(df$avg[df$g == "a"], df$n_claims[df$g == "a"]), tolerance = 1e-6)
  expect_gt(pearson_dispersion(m), 0)
})
