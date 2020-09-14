package ca.ualberta.cs.hdbscancem;

import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

/**
 *
 * @author Michael Strobl
 */
public class CustomCovariance {

    private RealMatrix[] B_k;
    private double[] lambda_k;
    private int dim;
    private int k;
    private RealMatrix[] covariance_k;
    private RealMatrix[] W_k;
    private double[] n_k;

    public CustomCovariance(int dim, double[] n_k, RealMatrix[] W_k) {
        this.dim = dim;
        k = W_k.length;
        this.W_k = W_k;
        lambda_k = new double[k];
        covariance_k = new RealMatrix[k];
        this.n_k = n_k;
    }

    public void estimateVII() {
        for (int i = 0; i < k; i++) {
            lambda_k[i] = W_k[i].getTrace()/((double)n_k[i]*(double)dim);
            covariance_k[i] = MatrixUtils.createRealIdentityMatrix(dim).scalarMultiply(lambda_k[i]);
        }
    }

    public void estimateVVI() {
        RealMatrix[] diag_W_k = new RealMatrix[k];
        B_k = new RealMatrix[k];
        for (int i = 0; i < k; i++) {
            diag_W_k[i] = MatrixUtils.createRealIdentityMatrix(dim);
            for (int j = 0; j < dim; j++) {
                diag_W_k[i].setEntry(j,j,W_k[i].getEntry(j,j));
            }
        }

        for (int i = 0; i < k; i++) {
            double temp = Math.pow(new LUDecomposition(diag_W_k[i]).getDeterminant(),1.0/(double)dim);
            B_k[i] = diag_W_k[i].scalarMultiply(1.0/temp);
            lambda_k[i] = temp / (double)n_k[i];
        }


        for (int i = 0; i < k; i++) {
            covariance_k[i] = B_k[i].scalarMultiply(lambda_k[i]);
        }
    }

    public void estimateVVV() {
        for (int i = 0; i < k; i++) {
            covariance_k[i] = W_k[i].scalarMultiply(1.0/(double)n_k[i]);
        }
    }

    public RealMatrix[] getCovariance_k() {
        return covariance_k;
    }
}
