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

import org.apache.commons.math3.distribution.MixtureMultivariateNormalDistribution;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.distribution.fitting.MultivariateNormalMixtureExpectationMaximization;
import org.apache.commons.math3.exception.*;
import org.apache.commons.math3.exception.util.LocalizedFormats;
import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.util.MathArrays;
import org.apache.commons.math3.util.Pair;

import java.util.List;

/**
 *
 * @author Michael Strobl
 */
public class MultivariateNormalClassificationExpectationMaximization extends MultivariateNormalMixtureExpectationMaximization {
    /**
     * Default maximum number of iterations allowed per fitting process.
     */
    private static final int DEFAULT_MAX_ITERATIONS = 1000;
    /**
     * Default convergence threshold for fitting.
     */
    private static final double DEFAULT_THRESHOLD = 1E-5;
    /**
     * The data to fit.
     */
    private final double[][] data;
    /**
     * The model fit against the data.
     */
    private MixtureMultivariateNormalDistribution fittedModel;
    /**
     * The log likelihood of the data given the fitted model.
     */
    private double logLikelihood = 0d;

    private double classificationlogLikelihood = 0d;

    /**
     * Creates an object to fit a multivariate normal mixture model to data.
     *
     * @param data Data to use in fitting procedure
     * @throws NotStrictlyPositiveException if data has no rows
     * @throws DimensionMismatchException   if rows of data have different numbers
     *                                      of columns
     * @throws NumberIsTooSmallException    if the number of columns in the data is
     *                                      less than 2
     */
    public MultivariateNormalClassificationExpectationMaximization(double[][] data) throws NotStrictlyPositiveException, DimensionMismatchException, NumberIsTooSmallException {
        super(data);
        if (data.length < 1) {
            throw new NotStrictlyPositiveException(data.length);
        }

        this.data = new double[data.length][data[0].length];

        for (int i = 0; i < data.length; i++) {
            if (data[i].length != data[0].length) {
                // Jagged arrays not allowed
                throw new DimensionMismatchException(data[i].length,
                        data[0].length);
            }
            if (data[i].length < 2) {
                throw new NumberIsTooSmallException(LocalizedFormats.NUMBER_TOO_SMALL,
                        data[i].length, 2, true);
            }
            this.data[i] = MathArrays.copyOf(data[i], data[i].length);
        }
    }

    public void fit(final MixtureMultivariateNormalDistribution initialMixture,
                     final int maxIterations,
                     final double threshold,
                     String modelName)
            throws SingularMatrixException,
            NotStrictlyPositiveException,
            DimensionMismatchException {
        if (maxIterations < 1) {
            throw new NotStrictlyPositiveException(maxIterations);
        }

        if (threshold < Double.MIN_VALUE) {
            throw new NotStrictlyPositiveException(threshold);
        }

        final int n = data.length;

        // Number of data columns. Jagged data already rejected in constructor,
        // so we can assume the lengths of each row are equal.
        final int numCols = data[0].length;
        final int k = initialMixture.getComponents().size();

        final int numMeanColumns
                = initialMixture.getComponents().get(0).getSecond().getMeans().length;

        if (numMeanColumns != numCols) {
            throw new DimensionMismatchException(numMeanColumns, numCols);
        }

        int numIterations = 0;
        double previousLogLikelihood = 0d;

        logLikelihood = Double.NEGATIVE_INFINITY;

        // Initialize model to fit to initial mixture.
        fittedModel = new MixtureMultivariateNormalDistribution(initialMixture.getComponents());

        while (numIterations++ <= maxIterations &&
                FastMath.abs(previousLogLikelihood - logLikelihood) > threshold) {
            previousLogLikelihood = logLikelihood;
            double sumLogLikelihood = 0d;

            // Mixture components
            final List<Pair<Double, MultivariateNormalDistribution>> components
                    = fittedModel.getComponents();

            // Weight and distribution of each component
            final double[] weights = new double[k];

            final MultivariateNormalDistribution[] mvns = new MultivariateNormalDistribution[k];

            for (int j = 0; j < k; j++) {
                //weights[j] = components.get(j).getFirst();
                weights[j] = 1.0 / (double)components.size();
                mvns[j] = components.get(j).getSecond();
            }

            // E-step and C-step: compute the data dependent parameters of the expectation
            // function and assign each x to the cluster with the maximum posterior probability.
            // The percentage of row's total density between a row and a
            // component
            final double[][] c_ik = new double[n][k];
            final double[] n_k = new double[k];

            // Sum of gamma for each component
            //final double[] gammaSums = new double[k];

            // Sum of gamma times its row for each each component
            //final double[][] gammaDataProdSums = new double[k][numCols];
            final double[][] dataSums = new double[k][numCols];

            for (int i = 0; i < n; i++) {
                double bestDensity = 0.0;
                int best = 0;
                for (int j = 0; j < k; j++) {
                    double density = weights[j] * mvns[j].density(data[i]);
                    if (density > bestDensity) {
                        bestDensity = density;
                        best = j;
                    }
                }
                c_ik[i][best] = 1.0;
                n_k[best] += 1.0;

                for (int j = 0; j < numCols; j++) {
                    dataSums[best][j] += data[i][j];
                }
                sumLogLikelihood += Math.log(bestDensity);
            }

            logLikelihood = sumLogLikelihood / n;
            classificationlogLikelihood = sumLogLikelihood;

            //System.out.println("sum classification likelihood: " + sumLogLikelihood);

            // M-step: compute the new parameters based on the expectation
            // function.
            final double[] newWeights = new double[k];
            final double[][] newMeans = new double[k][numCols];

            for (int j = 0; j < k; j++) {
                newWeights[j] = n_k[j]/(double)n;
                for (int col = 0; col < numCols; col++) {
                    newMeans[j][col] = dataSums[j][col] / n_k[j];
                }
            }


            RealMatrix W = MatrixUtils.createRealMatrix(numCols,numCols);
            RealMatrix[] W_k = new RealMatrix[k];

            for (int i = 0; i < k; i++) {
                W_k[i] = MatrixUtils.createRealMatrix(numCols,numCols);
            }

            for (int j = 0; j < n; j++) {
                double[] o = data[j];
                for (int i = 0; i < k; i++) {
                    RealVector data = MatrixUtils.createRealVector(o);
                    RealVector mean = MatrixUtils.createRealVector(newMeans[i]);
                    RealVector first = data.subtract(mean);
                    RealMatrix m = first.outerProduct(first);
                    m = m.scalarMultiply(c_ik[j][i]);
                    W_k[i] = W_k[i].add(m);
                    W = W.add(m);
                }
            }

            CustomCovariance cv = new CustomCovariance(numCols, n_k, W_k);

            if (modelName.equals("VII")) {
                cv.estimateVII();
            } else if (modelName.equals("VVI")) {
                cv.estimateVVI();
            } else if (modelName.equals("VVV")) {
                cv.estimateVVV();
            } else {
                cv.estimateVVV();
            }

            RealMatrix[] covariance_k = cv.getCovariance_k();

            final double[][][] newCovMatArrays = new double[k][numCols][numCols];
            for (int j = 0; j < k; j++) {
                newCovMatArrays[j] = covariance_k[j].getData();
            }

            // Update current model
            fittedModel = new MixtureMultivariateNormalDistribution(newWeights,
                    newMeans,
                    newCovMatArrays);
        }

        if (FastMath.abs(previousLogLikelihood - logLikelihood) > threshold) {
            // Did not converge before the maximum number of iterations
            throw new ConvergenceException();
        }
    }

    /**
     * Gets the fitted model.
     *
     * @return fitted model or {@code null} if no fit has been performed yet.
     */
    public MixtureMultivariateNormalDistribution getFittedModel() {
        return new MixtureMultivariateNormalDistribution(fittedModel.getComponents());
    }

    public double getClassificationlogLikelihood() {
        return classificationlogLikelihood;
    }
}
