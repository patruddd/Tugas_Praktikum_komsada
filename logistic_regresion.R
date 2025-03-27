# Logistic Regression Methods: Newton-Raphson and IRLS

logistic_regression <- function(features, target, method = "newton", tol = 1e-6, max_iter = 1000) {
  n <- nrow(features)
  p <- ncol(features)
  coefficients <- rep(0, p)
  
  for (iter in 1:max_iter) {
    predicted_prob <- 1 / (1 + exp(-features %*% coefficients))
    
    if (method == "newton") {
      # Newton-Raphson Method
      gradient <- t(features) %*% (target - predicted_prob)
      hessian <- -t(features) %*% diag(as.vector(predicted_prob * (1 - predicted_prob))) %*% features
      coefficients_new <- coefficients - solve(hessian) %*% gradient
    } else if (method == "irls") {
      # Iteratively Reweighted Least Squares (IRLS)
      weights <- diag(as.vector(predicted_prob * (1 - predicted_prob)))
      gradient <- t(features) %*% (target - predicted_prob)
      hessian <- t(features) %*% weights %*% features
      coefficients_new <- coefficients + solve(hessian) %*% gradient
    } else {
      stop("Invalid method. Choose 'newton' or 'irls'.")
    }
    
    if (sum(abs(coefficients_new - coefficients)) < tol) {
      coefficients <- coefficients_new
      break
    }
    
    coefficients <- coefficients_new
  }
  
  list(
    method = ifelse(method == "newton", "Newton-Raphson", "IRLS"),
    coefficients = coefficients, 
    iterations = iter,
    fitted_probabilities = predicted_prob
  )
}

# Simulate data for testing
set.seed(42)
sample_size <- 100
predictor_matrix <- cbind(1, matrix(rnorm(sample_size * 2), nrow = sample_size))
true_coefficients <- c(0.5, -1, 2)
binary_outcome <- rbinom(sample_size, 1, 1 / (1 + exp(-predictor_matrix %*% true_coefficients)))

# Run both methods
newton_result <- logistic_regression(predictor_matrix, binary_outcome, method = "newton")
irls_result <- logistic_regression(predictor_matrix, binary_outcome, method = "irls")

# Display results
print("Newton-Raphson Result:")
print(newton_result)

print("\nIRLS Result:")
print(irls_result)