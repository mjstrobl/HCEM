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

package ca.ualberta.cs;

import ca.ualberta.cs.hdbscancem.*;
import org.apache.commons.math3.distribution.MixtureMultivariateNormalDistribution;
import org.apache.commons.cli.*;
import java.io.*;
import java.util.*;

/**
 * @author Michael Strobl
 */
public class HCEM {

    public static double calculateARI(int[] labels, String[] trueLabels) {

        double a = .0;
        double b = .0;
        double c = .0;
        double d = .0;

        for (int i = 0; i < labels.length; i++) {
            for (int j = 0; j < labels.length; j++) {
                if (i != j) {
                    boolean sameClass = false;
                    boolean sameCluster = false;
                    if (labels[i] == labels[j]) {
                        sameCluster = true;
                    }
                    if (trueLabels[i].equals(trueLabels[j])) {
                        sameClass = true;
                    }
                    if (sameClass && sameCluster) {
                        a++;
                    }
                    if (sameClass && !sameCluster) {
                        b++;
                    }
                    if (!sameClass && sameCluster) {
                        c++;
                    }
                    if (!sameClass && !sameCluster) {
                        d++;
                    }
                }
            }
        }
        double m = a + b + c + d;
        double ari = (a - (((a + c) * (a + b)) / m)) / (((a + c + a + b) / 2) - (((a + c) * (a + b)) / m));

        return ari;
    }

    public static double evaluateDistributions(MixtureMultivariateNormalDistribution trueDistribution, MixtureMultivariateNormalDistribution distribution) {
        double klDiv = 0.0;
        int samples = 100000;
        for (int i = 0; i < samples; ++i) {
            double[] sample = trueDistribution.sample();
            double trueDensity = trueDistribution.density(sample);
            double density = distribution.density(sample);


            klDiv = klDiv + Math.log(trueDensity) - Math.log(density);
        }

        klDiv /= samples;

        return klDiv;
    }


    public static void clusterFile(String filename, String separator, boolean groundTruthLabels, boolean groundTruthDistribution, String criterion, int[] minPoints) throws IOException {
        Map<String, Object> bestResult = null;
        double maxCriterion = Double.NEGATIVE_INFINITY;
        int bestMinPoints = -1;

        System.out.println("Filename: " + filename + "; Criterion: " + criterion);
        DataReader dr = new DataReader(filename, separator);
        dr.readFile(groundTruthLabels, groundTruthDistribution);

        long startTime = System.currentTimeMillis();
        for (int i = 0; i < minPoints.length; i++) {
            String hierarchyFilename = HDBSCANRunner.startHDBSCAN(criterion, filename, dr.getDataset(), minPoints[i]);
            HDBSCANICRunner runner = new HDBSCANICRunner();

            List<int[]> hierarchy = runner.readCompactHierarchy(hierarchyFilename);
            Map<Integer, Cluster> clusters = runner.processHierarchy(hierarchy);
            List<Integer> initialPartition = runner.createInitialDistributions(clusters, dr.getDataset());
            Map<String, List<Integer>> candidateSolutions = runner.createCandidateSolutions(initialPartition, clusters, dr.getDataset());
            Map<String, Object> result = null;
            if (candidateSolutions != null && candidateSolutions.size() > 0) {
                result = runner.createFinalDistribution(candidateSolutions, dr.getDataset(), dr.getTrainingset(), dr.getTestset(), clusters, criterion);
            }

            if (result != null && (double) result.get("criterion") > maxCriterion) {
                maxCriterion = (double) result.get("criterion");
                bestResult = result;
                bestMinPoints = minPoints[i];
            }
        }
        double runTime = ((new Date()).getTime() - startTime) / 1000.0;
        System.out.println("Runtime: " + runTime);

        if (bestResult == null) {
            System.out.println("Could not cluster file.");
            return;
        }
        MixtureMultivariateNormalDistribution distribution = (MixtureMultivariateNormalDistribution) bestResult.get("distribution");
        String foundModel = (String) bestResult.get("model");
        int[] clustering = (int[]) bestResult.get("clusters");

        System.out.println("MinPoints: " + bestMinPoints + ", model: " + foundModel + ", components: " + distribution.getComponents().size() + ", max criterion value: " + maxCriterion);

        if (groundTruthDistribution) {
            double kld = evaluateDistributions(dr.getTrueDistribution(), distribution);
            System.out.println("KLD: " + kld);
        }

        if (groundTruthLabels) {
            double ari = calculateARI(clustering, dr.getLabels());
            System.out.println("ARI: " + ari);
        }
    }

    public static void main(String[] args) {

        String separator = " ";
        int[] minPts = {5,10};
        boolean groundTruthLabelsAvailable = false;
        boolean groundTruthDistributionAvailable = false;

        Options options = new Options();

        Option input = new Option("f", "filename", true, "input file path");
        input.setRequired(true);
        options.addOption(input);

        Option criterion = new Option("c", "criterion", true, "criterion");
        criterion.setRequired(true);
        options.addOption(criterion);

        Option groundTruthLabels = new Option("l", "labels", false, "labels available");
        groundTruthLabels.setRequired(false);
        options.addOption(groundTruthLabels);

        Option groundTruthDistribution = new Option("d", "distribution", false, "distribution available");
        groundTruthDistribution.setRequired(false);
        options.addOption(groundTruthDistribution);

        CommandLineParser parser = new DefaultParser();
        HelpFormatter formatter = new HelpFormatter();
        CommandLine cmd = null;

        try {
            cmd = parser.parse(options, args);
        } catch (ParseException e) {
            System.out.println(e.getMessage());
            formatter.printHelp("utility-name", options);

            System.exit(1);
        }

        String inputFilePath = cmd.getOptionValue("filename");
        String criterionName = cmd.getOptionValue("criterion");

        if (cmd.hasOption("labels")) {
            groundTruthLabelsAvailable = true;
        }

        if (cmd.hasOption("distribution")) {
            groundTruthDistributionAvailable = true;
        }

        try {
            clusterFile(inputFilePath,separator,groundTruthLabelsAvailable,groundTruthDistributionAvailable,criterionName,minPts);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
