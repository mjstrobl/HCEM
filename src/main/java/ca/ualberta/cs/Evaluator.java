package ca.ualberta.cs;

import org.apache.commons.math3.distribution.MixtureMultivariateNormalDistribution;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.linear.SingularMatrixException;
import org.apache.commons.math3.util.Pair;

import java.io.*;
import java.text.DecimalFormat;
import java.util.*;

public class Evaluator {


    public static MixtureMultivariateNormalDistribution readEstimatedDistributionFile(String filename) throws IOException, SingularMatrixException {

        File file = new File(filename);
        FileInputStream fis = new FileInputStream(file);
        byte[] data = new byte[(int) file.length()];
        fis.read(data);
        fis.close();

        String str = new String(data, "UTF-8");

        if (str.contains("nan")) {
            throw new IOException();
        }



        double[][] means = null;
        double[][][] covariances = null;
        double[] weights = null;
        try(BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line = br.readLine();
            int index = 0;
            int sigmaIndex = 0;
            boolean mean = false;
            boolean sigma = false;
            boolean weight = false;
            while (line != null) {
                if (line.equals("weights")) {
                    weight = true;
                    index = 0;
                } else if (line.equals("means")) {
                    mean = true;
                    weight = false;
                    index = 0;
                } else if (line.equals("covariances")) {
                    sigma = true;
                    mean = false;
                    weight = false;
                    index = 0;
                } else {
                    if (weight) {
                        weights = Arrays.stream(line.split(",")).mapToDouble(Double::parseDouble).toArray();
                    } else if (mean) {
                        double[] values = Arrays.stream(line.split(",")).mapToDouble(Double::parseDouble).toArray();
                        if (index == 0) {
                            means = new double[weights.length][values.length];
                        }
                        means[index++] = values;
                    } else if (sigma) {
                        double[] values = Arrays.stream(line.split(",")).mapToDouble(Double::parseDouble).toArray();
                        if (index == 0 && sigmaIndex == 0) {
                            covariances = new double[weights.length][values.length][values.length];
                        }

                        covariances[index][sigmaIndex] = values;
                        sigmaIndex++;
                        if (sigmaIndex == values.length) {
                            sigmaIndex = 0;
                            index++;
                        }
                    }
                }
                line = br.readLine();
            }
        }

        return new MixtureMultivariateNormalDistribution(weights, means, covariances);
    }

    public static Map<String,Object> readTrueDistributionFile(String filename, int clusters, int dimensions) {
        String separator = " ";
        try {
            double[][] means = new double[clusters][dimensions];
            double[][][] covariances = new double[clusters][dimensions][dimensions];
            double[] weights = new double[clusters];

            List<double[]> dataset = new ArrayList<>();
            List<String> labels = new ArrayList<>();

            try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
                String line = br.readLine();
                int index = 0;
                int sigmaIndex = 0;
                boolean mean = false;
                boolean sigma = false;
                boolean weight = false;
                while (line != null) {
                    if (line.equals("# Means")) {
                        mean = true;
                        index = 0;
                    } else if (line.equals("# Weights")) {
                        mean = false;
                        weight = true;
                        index = 0;
                    } else if (line.equals("# Sigmas")) {
                        sigma = true;
                        mean = false;
                        weight = false;
                        index = 0;
                    } else if (line.equals("# Noise")) {
                        mean = false;
                        sigma = false;
                        weight = false;
                    } else {
                        if (mean) {
                            String[] parts = line.split(separator);
                            for (int i = 0; i < parts.length; i++) {
                                means[index][i] = Double.parseDouble(parts[i]);
                            }
                            index++;
                        } else if (weight) {
                            weights[index++] = Double.parseDouble(line);
                        } else if (sigma) {
                            String[] parts = line.split(separator);
                            for (int i = 0; i < parts.length; i++) {
                                covariances[index][sigmaIndex][i] = Double.parseDouble(parts[i]);
                            }
                            sigmaIndex++;
                            if (sigmaIndex == dimensions) {
                                sigmaIndex = 0;
                                index++;
                            }
                        } else {
                            String[] tokens = line.split(" ");
                            String label = tokens[tokens.length-1];
                            labels.add(label);
                            tokens = Arrays.copyOfRange(tokens, 0, tokens.length-1);
                            double[] values = Arrays.stream(tokens).mapToDouble(Double::parseDouble).toArray();
                            dataset.add(values);
                        }
                    }
                    line = br.readLine();
                }
            }
            Map<String,Object> data = new HashMap<>();
            data.put("labels",labels);
            data.put("dataset",dataset);
            data.put("distribution",new MixtureMultivariateNormalDistribution(weights, means, covariances));

            return data;
        } catch (IOException ex) {
            return null;
        }
    }

    public static double computeKLD(MixtureMultivariateNormalDistribution trueDistribution, MixtureMultivariateNormalDistribution estimatedDistribution, double[][] samples) {
        double klDiv = 0.0;
        for (int i = 0; i < samples.length; i++) {
            double[] sample = samples[i];
            double trueDensity = trueDistribution.density(sample);
            double density = estimatedDistribution.density(sample);

            klDiv = klDiv + Math.log(trueDensity) - Math.log(density);
        }

        klDiv /= (double)samples.length;

        return klDiv;
    }

    public static double computeARI(int[] labels, List<String> trueLabels) {

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
                    if (trueLabels.get(i).equals(trueLabels.get(j))) {
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
        double m = a+b+c+d;
        double ari = ( a - ((( a + c ) * ( a + b )) / m )) / ((( a + c + a + b ) / 2 ) - ((( a + c ) * ( a + b )) / m ));

        return ari;
    }

    public static int[] createClustering(MixtureMultivariateNormalDistribution distribution, List<double[]> dataset) {
        int[] clustering = new int[dataset.size()];

        List<Pair<Double, MultivariateNormalDistribution>> components = distribution.getComponents();

        for (int i = 0; i < dataset.size(); i++) {
            int bestCluster = 0;
            double bestProb = 0.0;
            for (int k = 0; k < components.size(); k++) {
                double prob = components.get(k).getFirst()*components.get(k).getSecond().density(dataset.get(i));
                if (prob > bestProb) {
                    bestProb = prob;
                    bestCluster = k;
                }
            }
            clustering[i] = bestCluster;
        }

        return clustering;
    }



    public static void evaluateAll() throws IOException {
        //String[] models = {"VII","VVV"};
        int[] dimensions = {2,5,10,25,50};
       // int[] dimensions = {2};
        int[] clusters = {4,10,40};
        int numDatasets = 20;
        String[] criteria = {"BIC","ICL","NEC","VAL"};
        //String[] criteria = {"BIC","ICL","VAL"};

        String mixmodPath = "/home/michi/PycharmProjects/mixmod_HCEM/new_results/mixmod_";
        String mclustPath = "/home/michi/new_mclust_results/mclust_";
        String hcemPath = "/home/michi/new_hcem_results/hcem_";

        Map<String,String> paths = new HashMap<>();
        paths.put("Mclust",mclustPath);
        paths.put("Mixmod",mixmodPath);
        paths.put("HCEM",hcemPath);

        DecimalFormat df = new DecimalFormat("0.00");

        Map<String,double[][]> resultsARI = new HashMap<>();
        Map<String,double[][]> resultsKLD = new HashMap<>();

        Map<String,Set<Integer>> ignore = new HashMap<>();
        for (String algorithm : paths.keySet()) {
            for (String criterion : criteria) {
                String alg = algorithm + "(" + criterion + ")";
                ignore.put(alg, new HashSet<>());
            }
        }

        String outputFilename = "/home/michi/evaluator_all.txt";
        FileWriter myWriter = new FileWriter(outputFilename);


        for (int j = 0; j < dimensions.length; j++) {
            int d = dimensions[j];
            for (int k = 0; k < clusters.length; k++) {
                int c = clusters[k];
                for (int i = 0; i < numDatasets; i++) {

                    String dataFilename = "/home/michi/new_datasets/" + d + "d-" + c + "c-min" + (5 * d) + "-max" + (10 * d) + "/no" + i + ".dat";
                    Map<String,Object> data = readTrueDistributionFile(dataFilename,c,d);
                    MixtureMultivariateNormalDistribution trueDistribution = (MixtureMultivariateNormalDistribution)data.get("distribution");
                    List<double[]> dataset = (List<double[]>)data.get("dataset");
                    List<String> labels = (List<String>)data.get("labels");

                    int numSamples = 10000;
                    double[][] samples = new double[numSamples][d];
                    for (int s = 0; s < numSamples; s++) {
                        samples[s] = trueDistribution.sample();
                    }

                    for (String algorithm : paths.keySet()) {
                        String path = paths.get(algorithm);
                        for (String criterion : criteria) {
                            String alg = algorithm + "(" + criterion + ")";
                            String filename = path + criterion + "_" + d + "d-" + c + "c-min" + (5 * d) + "-max" + (10 * d) + "_no" + i + ".result";
                            int index = j*clusters.length + k;
                            try {
                                MixtureMultivariateNormalDistribution estimatedDistribution = readEstimatedDistributionFile(filename);
                                int[] clustering = createClustering(estimatedDistribution,dataset);
                                double kld = computeKLD(trueDistribution,estimatedDistribution,samples);
                                double ari = computeARI(clustering,labels);

                                System.out.println(d + "d " + c + "c " + "no" + i + " " + algorithm + " " + criterion);
                                System.out.println("KLD: " + df.format(kld));
                                System.out.println("ARI: " + df.format(ari));

                                myWriter.write(alg + "_" + d + "_" + c + "_" + i + "_" + ari + "_" + kld + "\n");



                                if (!resultsARI.containsKey(alg)) {
                                    resultsARI.put(alg,new double[dimensions.length*clusters.length][numDatasets]);
                                    resultsKLD.put(alg,new double[dimensions.length*clusters.length][numDatasets]);
                                }
                                resultsARI.get(alg)[index][i] = ari;
                                resultsKLD.get(alg)[index][i] = kld;
                            } catch (IOException e) {
                                System.out.println("file not found: " + filename);
                                ignore.get(alg).add(index);
                            } catch (SingularMatrixException e) {
                                System.out.println("exception happened with file: " + filename);
                                ignore.get(alg).add(index);
                            }
                        }
                    }
                }
            }
        }
        myWriter.close();
        System.out.println("ARI:");
        createString(resultsARI,df,numDatasets, ignore);
        System.out.println("\n\n\nKLD:");
        createString(resultsKLD,df,numDatasets, ignore);


    }

    public static void createTables() throws IOException {


        String outputFilename = "/home/michi/evaluator_all.txt";
        BufferedReader reader = new BufferedReader(new FileReader(outputFilename));
        String line = reader.readLine();
        Map<String,List<Double>> results = new HashMap<>();
        while (line != null) {
            //System.out.println(line);
            String[] tokens = line.split("_");
            if (tokens.length == 6) {
                String algorithm = tokens[0];
                int dim = Integer.parseInt(tokens[1]);
                int c = Integer.parseInt(tokens[2]);
                int n = Integer.parseInt(tokens[3]);
                double ari = Double.parseDouble(tokens[4]);
                double kld = Double.parseDouble(tokens[5]);
                if (dim != 100) {
                    if (!results.containsKey(algorithm)) {
                        results.put(algorithm,new ArrayList<>());
                    }
                    results.get(algorithm).add(ari);
                }

            }

            line = reader.readLine();
        }
        reader.close();

        generateCIs(results,new DecimalFormat("0.00"));
    }


    public static double calculateParams(double average, double std, int n) {

        double difference = 1.96 * ( std / Math.sqrt(n) );
        return difference;
    }

    public static double computeSTD(double mean, double[] values) {

        double std = 0.0;

        for (double value : values) {
            std += Math.pow(value - mean, 2);
        }

        return Math.sqrt(std / values.length);
    }

    public static double computeSTD(double mean, List<Double> values) {

        double std = 0.0;

        for (double value : values) {
            std += Math.pow(value - mean, 2);
        }

        return Math.sqrt(std / values.size());
    }

    private static void generateCIs(Map<String,List<Double>> results, DecimalFormat df) {

        String[] algorithms = {"Mclust(BIC)","Mixmod(BIC)","Mixmod(ICL)","Mixmod(NEC)","HCEM(BIC)","HCEM(ICL)","HCEM(VAL)"};

        for (String algorithm : algorithms) {
            int counter = 0;
            double overallAvg = 0.0;

            List<Double> values = results.get(algorithm);

            for (int i = 0; i < values.size(); i++) {
                double value = values.get(i);
                overallAvg += value;
            }

            overallAvg /= (double)values.size();
            double overallAvgStd = computeSTD(overallAvg,values);
            double confidence = calculateParams(overallAvg,overallAvgStd,values.size());

            String result = algorithm + ": " + df.format(overallAvg) + " +- " + df.format(confidence);
            System.out.println(result);
        }

    }

    private static void createString(Map<String,double[][]> resultsMap, DecimalFormat df, int numDatasets, Map<String,Set<Integer>> ignore) {
        StringBuilder latex_avg = new StringBuilder();
        StringBuilder latex_diff = new StringBuilder();

        //String[] algorithms = {"Mclust(BIC)","Mixmod(BIC)","Mixmod(ICL)","Mixmod(NEC)","HCEM(BIC)","HCEM(ICL)","HCEM(VAL)"};
        String[] algorithms = {"Mixmod(BIC)","HCEM(BIC)"};

        for (String algorithm : algorithms) {
            int counter = 0;
            double overallAvg = 0.0;

            double[][] results = resultsMap.get(algorithm);
            double[] avgStd = new double[results.length*results[0].length];
            latex_avg.append(algorithm);
            latex_diff.append(algorithm);
            for (int i = 0; i < results.length; i++) {
                double accumulated = 0.0;
                for (int j = 0; j < numDatasets; j++) {
                    accumulated += results[i][j];
                    if (i < avgStd.length) {
                        overallAvg += results[i][j];
                        avgStd[counter] = results[i][j];
                        counter += 1;
                    }
                }
                double result = accumulated / (double)numDatasets;
                double std = computeSTD(result,results[i]);
                double confidence = calculateParams(result,std,results[i].length);
                if (ignore.get(algorithm).contains(i)) {
                    latex_avg.append(" & ");
                    latex_diff.append(" & ");
                } else {
                    //latex_avg.append(" & ").append(df.format(result)).append("$\\pm$").append(df.format(confidence));
                    latex_avg.append(" & ").append(df.format(result));
                    latex_diff.append(" & ").append(df.format(confidence));
                }
            }

            overallAvg /= (double)counter;
            double overallAvgStd = computeSTD(overallAvg,avgStd);
            double confidence = calculateParams(overallAvg,overallAvgStd,avgStd.length);
            latex_avg.append(" & ").append(df.format(overallAvg)).append("$\\pm$").append(df.format(confidence)).append(" \\\\ \n");
            latex_diff.append(" & " + df.format(confidence) + " \\\\ \n");
        }
        latex_avg.append(" \\hline\n");
        latex_diff.append(" \\hline\n");

        System.out.println(latex_avg);
        System.out.println("\n");
        System.out.println(latex_diff);

    }

    public static void main(String[] args) {
        try {
            evaluateAll();
            //createTables();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
