# Download the freMTPL2 frequency and severity tables from the CASdatasets
# GitHub repository (Dutang & Charpentier, GPL >= 2) and convert them to CSV.
#
# Usage (from the project root):  Rscript data/download_data.R
# Output: data/raw/freMTPL2freq.csv, data/raw/freMTPL2sev.csv
# Only base R is required.

base_url <- "https://raw.githubusercontent.com/dutangc/CASdatasets/master/data"
raw_dir <- file.path("data", "raw")
dir.create(raw_dir, showWarnings = FALSE, recursive = TRUE)

fetch_and_convert <- function(name) {
  csv_path <- file.path(raw_dir, paste0(name, ".csv"))
  if (file.exists(csv_path)) {
    message("Already present: ", csv_path)
    return(invisible(csv_path))
  }
  rda_path <- file.path(raw_dir, paste0(name, ".rda"))
  url <- paste0(base_url, "/", name, ".rda")
  message("Downloading ", url)
  utils::download.file(url, rda_path, mode = "wb", quiet = TRUE)
  env <- new.env()
  load(rda_path, envir = env)
  df <- get(name, envir = env)
  # IDpol is stored as a factor in the .rda; write it as an integer key.
  df$IDpol <- as.integer(as.character(df$IDpol))
  utils::write.csv(df, csv_path, row.names = FALSE)
  message(sprintf("Wrote %s (%d rows, %d cols)", csv_path, nrow(df), ncol(df)))
  invisible(csv_path)
}

invisible(lapply(c("freMTPL2freq", "freMTPL2sev"), fetch_and_convert))
