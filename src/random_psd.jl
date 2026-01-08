using LinearAlgebra
using Random

"""
    random_matrix_with_eigenvalues(L::Real, mu::Real, n::Int)

Generates a random n x n symmetric matrix Q whose largest eigenvalue is L 
and smallest eigenvalue is mu.

Args:
- L: The largest eigenvalue.
- mu: The smallest eigenvalue.
- n: The dimension of the matrix.

Returns:
- Q: The random n x n symmetric matrix.
"""
function random_matrix_with_eigenvalues(L::Real, mu::Real, n::Int)
    
    # 1. Choose the Eigenvalues (Spectrum)
    # The first and last eigenvalues are fixed to L and mu.
    # The remaining n-2 eigenvalues are chosen randomly and uniformly 
    # between mu and L.
    
    if n < 2
        # Handle 1x1 case
        return fill(L, 1, 1) # A 1x1 matrix has only one eigenvalue, which must be L=mu.
    end

    if L < mu
        error("Largest eigenvalue (L) must be greater than or equal to smallest eigenvalue (mu).")
    end
    
    # Create the vector of eigenvalues
    eigenvalues = zeros(n)
    eigenvalues[1] = L
    eigenvalues[n] = mu
    
    # Fill the remaining eigenvalues (randomly distributed between mu and L)
    if n > 2
        # Use a uniform distribution for the intermediate eigenvalues.
        intermediate_eigs = rand(n - 2) * (L - mu) .+ mu
        # Sort them to maintain a clear order in the diagonal matrix (optional, but good practice)
        sort!(intermediate_eigs, rev=true) 
        eigenvalues[2:n-1] = intermediate_eigs
    end
    
    # Create the diagonal matrix Λ
    Lambda = Diagonal(eigenvalues)
    
    # 2. Generate a Random Orthogonal Matrix (V)
    # Use the QR decomposition of a random Gaussian matrix to get a random orthogonal matrix.
    # This ensures the eigenvectors are chosen randomly (isotropic).
    
    # Create a random matrix A with standard normal entries
    A = randn(n, n)
    
    # Perform QR decomposition: A = V * R
    # The Q factor is the random orthogonal matrix V
    V = qr(A).Q
    
    # 3. Construct the Random Matrix Q
    # Q = V * Λ * V'
    Q = V * Lambda * V'
    
    # The result should be symmetric, but due to floating-point arithmetic, 
    # explicitly making it symmetric ensures perfect symmetry.
    return Symmetric(Q)
end
