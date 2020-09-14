/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
