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
import org.apache.commons.math3.util.Pair;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/**
 *
 * @author Michael Strobl
 */
public class HDBSCANICRunner {

    private String[] modelNames = {"VVV","VVI","VII"};
    //private String[] modelNames = {"VVV","VII"};
    //private String modelName;
    private static final double MIN_LOGLIKELIHOOD = -100000;

    /*public HDBSCANICRunner (String model) {
        this.modelName = model;
    }*/

    /**
     * Read hierarchy file produced by HDBSCAN.
     * @param filename
     * @return List of cluster splits.
     * @throws IOException
     */
    public List<int[]> readCompactHierarchy(String filename) throws IOException {
        List<int[]> hierarchy = new ArrayList<>();
        try(BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line = br.readLine();
            while (line != null) {
                String[] tokens = line.split(",");
                int[] objects = Arrays.asList(tokens).subList(1, tokens.length).stream().mapToInt(Integer::parseInt).toArray();
                hierarchy.add(objects);
                line = br.readLine();
            }
        }

        return hierarchy;
    }

    /**
     * Process hierarchy.
     * @param hierarchy List from hierarchy file.
     * @return Map of clusters and their ids.
     */
    public Map<Integer,Cluster> processHierarchy(List<int[]> hierarchy) {
        Map<Integer,Cluster> clusters = new HashMap<>();
        int[] previousLabels = new int[hierarchy.get(0).length];
        for (int i = 0; i < hierarchy.size(); i++) {
            int[] objects = hierarchy.get(i);
            for (int j = 0; j < objects.length; j++) {
                int currentCluster = objects[j];
                if (currentCluster == 0 || currentCluster == previousLabels[j]) {
                    continue;
                }
                if (!clusters.containsKey(currentCluster)) {
                    if (previousLabels[j] > 0) {
                        clusters.get(previousLabels[j]).addChild(currentCluster);
                    }
                    Cluster c = new Cluster(currentCluster,previousLabels[j]);
                    clusters.put(currentCluster,c);
                }
                clusters.get(currentCluster).getMembers().add(j);
                previousLabels[j] = currentCluster;
            }
        }

        return clusters;
    }


    /**
     * Create initial distribution from clustering hierarchy leaves.
     * @param clusters
     * @param dataset
     */
    public List<Integer> createInitialDistributions(Map<Integer,Cluster> clusters, double[][] dataset) {

        //Find all leaves in the hierarchy.
        List<Integer> leaves = new ArrayList<>();
        for (int id : clusters.keySet()) {
            if (!clusters.get(id).hasChildren()) {
                leaves.add(id);
            }
        }

        //System.out.println("Number of leaves: " + leaves.size());

        List<Integer> currentPartition = leaves;
        String currentModelName = "no model";

        //Loop over all model names and find distribution with the maximum number of components given an HDBSCAN clustering hierarchy.
        while (currentPartition.size() > 1) {
            List<Integer> notWorking = new ArrayList<>();
            for (String modelName : modelNames) {
	    	notWorking = new ArrayList<>();
            	//Try to create mixture model from current partition (leaves initially).
            	for (int cluster : currentPartition) {
                    try {
                        clusters.get(cluster).createComponent(dataset,modelName);
                    } catch (Exception ex) {
                        notWorking.add(cluster);
                    }
                }
                if (notWorking.size() == 0) {
                    //All Gaussians estimated.
                    currentModelName = modelName;
                    break;
                }
	    }
            if (notWorking.size() == 0) {
                break;
            } else {
                //If not possible to create mixture model, combine siblings and try again with a different model.
                currentPartition = combineSiblings(currentPartition, clusters, notWorking);
            }
        }

        if (currentPartition.size() == 1) {
            //Clustering failed, Gaussians could not be estimated.
            return null;
        } else {
            //Initial model created, assign all objects to current partition.
            assignObjectsToInitialPartition(currentPartition,clusters,dataset,currentModelName);
            return currentPartition;
        }

    }

    public Set<Integer> getDescendantsOfParent(Map<Integer,Cluster> clusters,int parent) {

        Set<Integer> descendants = new HashSet<>(clusters.get(parent).getChildren());
        while (true) {
            int previousSize = descendants.size();
            Set<Integer> moreDescendants = new HashSet<>(descendants);
            for (int descendant : descendants) {
                Set<Integer> children = clusters.get(descendant).getTempChildren();
                moreDescendants.addAll(children);
            }
            descendants = moreDescendants;
            int currentSize = descendants.size();
            if (currentSize == previousSize) {
                break;
            }
        }

        return descendants;
    }

    /**
     * Combine siblings in case it was not possible to create a mixture model with the current partition.
     * @param currentPartition
     * @param clusters
     * @param notWorking
     * @return
     */
    public List<Integer> combineSiblings(List<Integer> currentPartition, Map<Integer,Cluster> clusters,List<Integer> notWorking) {

        Set<Integer> parentsToAdd = new HashSet<>();
        Set<Integer> childrenToRemove = new HashSet<>();
        for (int child : notWorking) {
            int parent = clusters.get(child).getParent();
            Set<Integer> descendants = getDescendantsOfParent(clusters, parent);
            parentsToAdd.add(parent);
            childrenToRemove.addAll(descendants);
        }

        List<Integer> newPartition = new ArrayList<>(currentPartition);
        newPartition.addAll(parentsToAdd);
        newPartition.removeAll(childrenToRemove);

        return newPartition;
    }

    /**
     * Assign all objects in the given dataset to clusters and re-estimate model parameters.
     * @param currentPartition
     * @param clusters
     * @param dataset
     */
    private void assignObjectsToInitialPartition(List<Integer> currentPartition, Map<Integer,Cluster> clusters, double[][] dataset, String modelName) {
        int k = currentPartition.size();

        List<List<Integer>> indices_k = new ArrayList<>();
        for (int i = 0; i < k; i++) {
            indices_k.add(new ArrayList<>());
        }

        for (int i = 0; i < dataset.length; i++) {
            double[] o = dataset[i];
            int bestCluster = 0;
            double bestDensity = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < k; j++) {
                double density = clusters.get(currentPartition.get(j)).getDistributionPair(modelName).getSecond().density(o);
                if (density > bestDensity) {
                    bestDensity = density;
                    bestCluster = j;
                }
            }
            indices_k.get(bestCluster).add(i);
        }

        for (int i = 0; i < k; i++) {
            List<Integer> indices = new ArrayList<>(indices_k.get(i).size());
            for (int j = 0; j < indices_k.get(i).size(); j++) {
                indices.add(indices_k.get(i).get(j));
            }
            clusters.get(currentPartition.get(i)).setMembers(indices);
        }

        List<Pair<Double,MultivariateNormalDistribution>> components = new ArrayList<>();
        for (int cluster : currentPartition) {
            clusters.get(cluster).createComponent(dataset,modelName);
            components.add(clusters.get(cluster).getDistributionPair(modelName));
        }

        //MultivariateNormalClassificationExpectationMaximization mnmem = new MultivariateNormalClassificationExpectationMaximization(dataset);
        //MixtureMultivariateNormalDistribution fittedDistribution = new MixtureMultivariateNormalDistribution(components);
        //mnmem.fit(fittedDistribution, 100, 1E-2,modelName);
        //components = mnmem.getFittedModel().getComponents();

        for (int i = 0; i < currentPartition.size(); i++) {
            Cluster cluster = clusters.get(currentPartition.get(i));
            cluster.setDistributionPair(modelName,components.get(i));
            cluster.setMembers(new ArrayList<>());
        }

        for (int i = 0; i < dataset.length; i++) {
            int bestCluster = 0;
            double bestProb = 0.0;
            for (k = 0; k < currentPartition.size(); k++) {
                Cluster cluster = clusters.get(currentPartition.get(k));
                double prob = cluster.getDistributionPair(modelName).getFirst()*cluster.getDistributionPair(modelName).getSecond().density(dataset[i]);
                if (prob > bestProb) {
                    bestProb = prob;
                    bestCluster = currentPartition.get(k);
                }
            }
            clusters.get(bestCluster).addMember(i);
        }
    }



    /**
     * Create candidate solutions, one for each model.
     * @param initialPartition
     * @param clusters
     * @param dataset
     * @return
     */
    public Map<String,List<Integer>> createCandidateSolutions(List<Integer> initialPartition, Map<Integer,Cluster> clusters, double[][] dataset) {
        Map<String, List<Integer>> candidateSolutions = new HashMap<>();
        List<Integer> bestSolution = null;
        double overallBestCL = 0.0;
	for (String modelName : modelNames) {
        try {
            List<Integer> currentPartition = new ArrayList<>(initialPartition);

            for (int c : currentPartition) {
                clusters.get(c).createComponent(dataset, modelName);
            }

            overallBestCL = computeClassificationLikelihood(clusters, currentPartition, dataset, modelName);
            bestSolution = currentPartition;
            while (true) {
                List<Integer> parents = new ArrayList<>();

                for (int id : currentPartition) {
                    int parent = clusters.get(id).getTempParent();
                    if (parent > 1 && !parents.contains(parent)) {
                        parents.add(parent);
                    }
                }

                if (parents.isEmpty()) {
                    break;
                }

                Collections.sort(parents);

                for (int j = parents.size() - 1; j >= 0; j--) {
                    int lastParent = parents.get(j);
                    Set<Integer> children = clusters.get(lastParent).getTempChildren();

                    List<Integer> newMembers = new ArrayList<>();
                    for (int child : children) {
                        Cluster c = clusters.get(child);
                        newMembers.addAll(c.getMembers());
                    }

                    clusters.get(lastParent).setMembers(newMembers);

                    boolean allInCurrentSolution = true;

                    for (int child : children) {
                        if (!currentPartition.contains(child)) {
                            allInCurrentSolution = false;
                            break;
                        }
                    }

                    if (allInCurrentSolution) {
                        Set<Integer> parentPartition = new HashSet<>(currentPartition);
                        parentPartition.removeAll(children);
                        parentPartition.add(lastParent);

                        List<Integer> parentPartitionList = new ArrayList<>(parentPartition);
                        clusters.get(lastParent).createComponent(dataset, modelName);

                        double parentCL = computeClassificationLikelihood(clusters, parentPartitionList, dataset, modelName);


                        if (parentCL > overallBestCL) {
                            overallBestCL = parentCL;
                            bestSolution = parentPartitionList;
                            currentPartition = parentPartitionList;

                        } else {
                            int grandParent = clusters.get(lastParent).getTempParent();
                            clusters.get(grandParent).getTempChildren().remove(lastParent);
                            clusters.get(grandParent).getTempChildren().addAll(children);
                            for (int child : children) {
                                clusters.get(child).setTempParent(grandParent);
                            }
                        }
                    }
                }
            }

        } catch (Exception ex) {

        }

        candidateSolutions.put(modelName, bestSolution);

        for (int key : clusters.keySet()) {
            Cluster c = clusters.get(key);
            c.setTempChildren(new HashSet<>(c.getChildren()));
            c.setTempParent(c.getParent());
        }
	}

        return candidateSolutions;
    }

    /**
     * Select final distributions with BIC (EM) and ICL (CEM).
     * @param candidateSolutions
     * @param dataset
     * @param clusters
     * @return
     */
    public Map<String,Object> createFinalDistribution(Map<String,List<Integer>> candidateSolutions, double[][] dataset, double[][][] trainingset, double[][][] testset, Map<Integer,Cluster> clusters, String criterion) {

        int dims = dataset[0].length;

        double maxCriterion = Double.NEGATIVE_INFINITY;
        String bestModel = "";
        MixtureMultivariateNormalDistribution bestDistribution = null;

        if (criterion.equals("BIC")) {
            for (String modelName : candidateSolutions.keySet()) {
                List<Integer> solution = candidateSolutions.get(modelName);
                if (solution == null) {
                    continue;
                }
                MixtureMultivariateNormalDistribution distribution = createDistribution(solution,clusters,modelName);
                if (distribution == null) {
                    continue;
                }

                MultivariateNormalLikelihoodExpectationMaximization mnmem = new MultivariateNormalLikelihoodExpectationMaximization(dataset);
                MixtureMultivariateNormalDistribution fittedDistribution = new MixtureMultivariateNormalDistribution(new ArrayList<>(distribution.getComponents()));
                double parameters = numberOfParameters(modelName,(double)dims,(double)fittedDistribution.getComponents().size());

                try {
                    mnmem.fit(fittedDistribution,10,1E-2,modelName);
                    double logLikelihood = mnmem.getNonclassificationlogLikelihood();
                    double BIC = logLikelihood - (parameters/2.0)*Math.log(dataset.length);
                    if (BIC > maxCriterion) {
                        maxCriterion = BIC;
                        bestModel = modelName;
                        bestDistribution = mnmem.getFittedModel();
                    }
                } catch (Exception ex) {
                    //ex.printStackTrace();
                }
            }
        } else if (criterion.equals("ICL")) {
            for (String modelName : candidateSolutions.keySet()) {
                List<Integer> solution = candidateSolutions.get(modelName);
                if (solution == null) {
                    continue;
                }
		        MixtureMultivariateNormalDistribution distribution = createDistribution(solution,clusters,modelName);
                if (distribution == null) {
                    continue;
                }

                MultivariateNormalClassificationExpectationMaximization mnmem = new MultivariateNormalClassificationExpectationMaximization(dataset);
                MixtureMultivariateNormalDistribution fittedDistribution = new MixtureMultivariateNormalDistribution(new ArrayList<>(distribution.getComponents()));
                double parameters = numberOfParameters(modelName,(double)dims,(double)fittedDistribution.getComponents().size());

                try {
                    mnmem.fit(fittedDistribution, 10, 1E-2, modelName);
                    double classificationLikelihood = mnmem.getClassificationlogLikelihood();
                    double ICL = classificationLikelihood - (parameters / 2.0) * Math.log(dataset.length);
                    if (ICL > maxCriterion) {
                        maxCriterion = ICL;
                        bestModel = modelName;
                        bestDistribution = mnmem.getFittedModel();
                    }
                } catch (Exception ex) {
                    //ex.printStackTrace();
                }
            }
        } else if (criterion.equals("VAL")) {
            for (String modelName : candidateSolutions.keySet()) {
                List<Integer> solution = candidateSolutions.get(modelName);
                if (solution == null) {
                    continue;
                }
		        MixtureMultivariateNormalDistribution distribution = createDistribution(solution,clusters,modelName);
                if (distribution == null) {
                    continue;
                }

                double avgL = 0.0;
                int n = 0;

                for (int i = 0; i < trainingset.length; i++) {
                    double[][] currentTrainingset = trainingset[i];
                    double[][] currentTestset = testset[i];

                    MultivariateNormalClassificationExpectationMaximization mnmem = new MultivariateNormalClassificationExpectationMaximization(currentTrainingset);
                    MixtureMultivariateNormalDistribution fittedDistribution = new MixtureMultivariateNormalDistribution(new ArrayList<>(distribution.getComponents()));

                    try {
                        mnmem.fit(fittedDistribution, 10, 1E-2, modelName);
                        //double L = computeLikelihood(clusters, solution, currentTestset, modelName);
                        double L = computeLikelihood(currentTestset,mnmem.getFittedModel());
                        avgL += L;
                        n++;
                        //System.out.println("model: " + modelName + ", i: " + i + ", L: " + L);
                    } catch (Exception ex) {
                        //ex.printStackTrace();
                    }
                }
                avgL /= n;
                //System.out.println("model: " + modelName + ", avg L: " + avgL + ", components: " + solution.size());
                if (avgL > maxCriterion) {
                    maxCriterion = avgL;
                    bestModel = modelName;
                    bestDistribution = distribution;

                }
            }
            if (bestDistribution != null) {
                MultivariateNormalClassificationExpectationMaximization mnmem = new MultivariateNormalClassificationExpectationMaximization(dataset);
                mnmem.fit(bestDistribution, 10, 1E-2, bestModel);
                bestDistribution = mnmem.getFittedModel();
            }
        }

        if (bestDistribution != null) {
            Map<String,Object> result = new HashMap<>();
            int[] clustering = createClustering(bestDistribution,dataset);
            result.put("model", bestModel);
            result.put("criterion",maxCriterion);
            result.put("distribution",bestDistribution);
            result.put("clusters",clustering);
            return result;
        } else {
            return null;
        }
    }


    private double computeClassificationLikelihood(Map<Integer,Cluster> clusters, List<Integer> partition, double[][] dataset,String modelName) {

        double CL = 0.0;
        for (int cluster : partition) {
            CL += clusters.get(cluster).computeLikelihood(modelName,dataset,MIN_LOGLIKELIHOOD);
        }

        return CL;
    }

    private double computeLikelihood(double[][] dataset,MixtureMultivariateNormalDistribution distribution) {

        double L = 0.0;

        for (int i = 0; i < dataset.length; i++) {
            double logP = 0.0;
            for (int k = 0; k < distribution.getComponents().size(); k++) {
                Pair<Double, MultivariateNormalDistribution> component = distribution.getComponents().get(k);
                double density = component.getSecond().density(dataset[i])*component.getFirst();
                logP += density;
            }
            logP = Math.max(Math.log(logP),MIN_LOGLIKELIHOOD);
            double log = (logP == logP) ? logP : 0.;
            L += log;
        }


        return L;
    }

    private double computeLikelihood(Map<Integer,Cluster> clusters, List<Integer> partition, double[][] dataset,String modelName) {

        double L = 0.0;

        for (int i = 0; i < dataset.length; i++) {
            double logP = 0.0;
            for (int k = 0; k < partition.size(); k++) {
                Pair<Double, MultivariateNormalDistribution> distribution = clusters.get(partition.get(k)).getDistributionPair(modelName);
                double density = distribution.getSecond().density(dataset[i])*distribution.getFirst();
                logP += density;
            }
            logP = Math.max(Math.log(logP),MIN_LOGLIKELIHOOD);
            double log = (logP == logP) ? logP : 0.;
            L += log;
        }


        return L;
    }

    private MixtureMultivariateNormalDistribution createDistribution(List<Integer> partition, Map<Integer,Cluster> clusters, String modelName) {
        List<Pair<Double, MultivariateNormalDistribution>> components = new ArrayList<>();
        for (int cluster : partition) {
            components.add(clusters.get(cluster).getDistributionPair(modelName));
        }


        try {
            MixtureMultivariateNormalDistribution distribution = new MixtureMultivariateNormalDistribution(components);


            return distribution;
        }catch (NullPointerException ex) {
            //ex.printStackTrace();
            return null;
        }
    }

    public int[] createClustering(MixtureMultivariateNormalDistribution distribution, double[][] dataset) {
        int[] clustering = new int[dataset.length];

        List<Pair<Double,MultivariateNormalDistribution>> components = distribution.getComponents();

        for (int i = 0; i < dataset.length; i++) {
            int bestCluster = 0;
            double bestProb = 0.0;
            for (int k = 0; k < components.size(); k++) {
                double prob = components.get(k).getFirst()*components.get(k).getSecond().density(dataset[i]);
                if (prob > bestProb) {
                    bestProb = prob;
                    bestCluster = k;
                }
            }
            clustering[i] = bestCluster;
        }

        return clustering;
    }

    private double numberOfParameters(String modelName, double dim, double k) {
        double alpha = k + dim + k - 1.0;
        double beta = dim * (dim + 1.0) / 2.0;
        if (modelName.equals("VVI")) {
            return alpha+k*dim;
        } else if (modelName.equals("VVV")) {
            return alpha+k*beta;
        } else if (modelName.equals("VII")) {
            return alpha+dim;
        }
        return 0.0;
    }
}
