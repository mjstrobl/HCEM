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

import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.SingularMatrixException;
import org.apache.commons.math3.util.Pair;

import java.util.*;

/**
 *
 * @author Michael Strobl
 */
public class Cluster {

    private int id;
    private int parent;
    private List<Integer> members;
    private Set<Integer> children;
    private Map<String,Pair<Double, MultivariateNormalDistribution>> distributionPairs;
    private Map<String,Double> completeLikelihoods;
    private int tempParent;
    private Set<Integer> tempChildren;

    public Cluster(int id, int parent) {
        this.id = id;
        this.parent = parent;
        this.tempParent = parent;
        this.members = new ArrayList<>();
        this.children = new HashSet<>();
        this.tempChildren = new HashSet<>();
        this.distributionPairs = new HashMap<>();
        this.completeLikelihoods = new HashMap<>();
    }

    public boolean hasChildren() {
        return children.size() > 0;
    }

    public void addMember(int member) {
        this.members.add(member);
    }

    public List<Integer> getMembers() {
        return members;
    }

    public void setMembers(List<Integer> members) {
        this.members = members;
    }

    public Set<Integer> getChildren() {
        return children;
    }

    public void addChild(int id) {
        this.children.add(id);
        this.tempChildren.add(id);
    }

    public int getParent() {
        return parent;
    }

    public void setParent(int parent) {
        this.parent = parent;
    }

    public int getId() {
        return id;
    }

    public Pair<Double, MultivariateNormalDistribution> getDistributionPair(String modelName) {
        if (distributionPairs.containsKey(modelName)) {
            return distributionPairs.get(modelName);
        } else {
            return null;
        }
    }

    public void setDistributionPair(String modelName, Pair<Double, MultivariateNormalDistribution> distributionPair) {
        distributionPairs.put(modelName,distributionPair);
    }

    public int getTempParent() {
        return tempParent;
    }

    public void setTempParent(int tempParent) {
        this.tempParent = tempParent;
    }

    public Set<Integer> getTempChildren() {
        return tempChildren;
    }

    public void setTempChildren(Set<Integer> tempChildren) {
        this.tempChildren = tempChildren;
    }

    public void createComponent(double[][] dataset, String modelName) {

        int dims = dataset[0].length;

        if (!distributionPairs.containsKey(modelName)) {
            double[] n_k = new double[1];
            n_k[0] = members.size();
            RealMatrix[] W_k = new RealMatrix[1];
            W_k[0] = MatrixUtils.createRealMatrix(dims, dims);

            double[][] currentDataset = new double[members.size()][dims];
            for (int j = 0; j < members.size(); j++) {
                currentDataset[j] = dataset[members.get(j)];
            }

            double[] means = new double[dims];
            double weight = (double)n_k[0] / (double)dataset.length;

            for (int j = 0; j < n_k[0]; j++) {
                for (int d = 0; d < dims; d++) {
                    means[d] += currentDataset[j][d];
                }
            }
            for (int d = 0; d < dims; d++) {
                means[d] /= (double) n_k[0];
            }

            for (int j = 0; j < currentDataset.length; j++) {
                double[] o = currentDataset[j];
                RealVector data = MatrixUtils.createRealVector(o);
                RealVector mean = MatrixUtils.createRealVector(means);
                RealVector first = data.subtract(mean);
                RealMatrix m = first.outerProduct(first);
                W_k[0] = W_k[0].add(m);
            }

            CustomCovariance cv = new CustomCovariance(dims, n_k, W_k);
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
            MultivariateNormalDistribution d = new MultivariateNormalDistribution(means, covariance_k[0].getData());
            Pair<Double, MultivariateNormalDistribution> distributionPair = new Pair<>(weight, d);
            setDistributionPair(modelName, distributionPair);
        }
    }


    public double computeLikelihood(String modelName,double[][] dataset, double minLikelihood) {
        if (completeLikelihoods.containsKey(modelName)) {
            return completeLikelihoods.get(modelName);
        } else {
            Pair<Double, MultivariateNormalDistribution> distribution = distributionPairs.get(modelName);
            double CL = 0.0;
            for (int i = 0; i < members.size(); i++) {
                double logP = Math.max(Math.log(distribution.getSecond().density(dataset[members.get(i)])*distribution.getFirst()),minLikelihood);
                double log = (logP == logP) ? logP : 0.;
                CL += log;
            }
            completeLikelihoods.put(modelName,CL);
            return CL;
        }
    }
}
