# Function to install packages with error handling
install_package_safely <- function(package_name, version = NULL, dependencies = TRUE) {
  tryCatch({
    if (!require("devtools", quietly = TRUE)) {
      install.packages("devtools", repos = "https://cran.rstudio.com/")
    }
    
    if (!is.null(version)) {
      devtools::install_version(package_name, version = version, dependencies = dependencies)
    } else {
      install.packages(package_name, dependencies = dependencies, repos = "https://cran.rstudio.com/")
    }
    
    cat(sprintf("Successfully installed %s\n", package_name))
  }, error = function(e) {
    cat(sprintf("Error installing %s: %s\n", package_name, e$message))
  })
}

# Install packages
install_package_safely("devtools")
install_package_safely("tidyverse", version = "1.3.1", dependencies = TRUE)
install_package_safely("lme4", version = "1.1-29", dependencies = TRUE)
install_package_safely("broom.mixed", version = "0.2.9.4", dependencies = NA)