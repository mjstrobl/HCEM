package ca.ualberta.cs;

import org.apache.commons.math3.distribution.MixtureMultivariateNormalDistribution;

import java.io.*;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;


/**
 *
 * @author Michael Strobl
 */
public class DataReader {
    //private double TRAININGSPLIT = 0.9;
    private int FOLDS = 10;
    private String filename;
    private String separator;
    private double[][] dataset;
    private double[][][] trainingset;
    private double[][][] testset;
    private String[] labels;
    private int dim;
    private int components;
    private MixtureMultivariateNormalDistribution trueDistribution;

    public DataReader(String filename, String separator) {
        this.filename = filename;
        this.separator = separator;
    }

    public void readFile(boolean groundTruthLabels, boolean groundTruthDistribution) throws IOException {
        int lines = 0;
        List<Integer> indices = new ArrayList<>();
        dim = 0;
        Set<String> components = new HashSet<>();
        try(BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line = br.readLine();
            String[] parts = line.split(separator);
            if (groundTruthLabels) {
                dim = parts.length - 1;
            } else {
                dim = parts.length;
            }
            while (line != null) {
                if (line.equals("# Means")) {
                    break;
                } else if (line.equals("# Sigma")) {
                    break;
                } else if (line.equals("# Noise")) {
                    break; //only if there's no noise.
                } else {
                    indices.add(lines);
                    lines++;
                }
                line = br.readLine();
            }
        }

        int foldSize = (int)(lines/FOLDS);

        testset = new double[FOLDS][foldSize][dim];
        dataset = new double[lines][dim];
        trainingset = new double[FOLDS][lines-foldSize][dim];
        labels = new String[lines];

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
                if (line.equals("# Means")) {
                    mean = true;
                    index = 0;
                    means = new double[components.size()][dim];
                } else if (line.equals("# Weights")) {
                    mean = false;
                    weight = true;
                    index = 0;
                    weights = new double[components.size()];
                }  else if (line.equals("# Sigmas")) {
                    sigma = true;
                    mean = false;
                    weight = false;
                    index = 0;
                    covariances = new double[components.size()][dim][dim];
                } else if (line.equals("# Noise")) {
                    mean = false;
                    sigma = false;
                    weight = false;
                } else {
                    if (groundTruthDistribution && groundTruthLabels && (mean || weight || sigma)) {
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
                            if (sigmaIndex == dim) {
                                sigmaIndex = 0;
                                index++;
                            }
                        }
                    } else {
                        String[] parts = line.split(separator);
                        for (int i = 0; i < parts.length; i++) {
                            if (i == parts.length - 1 && groundTruthLabels) {
                                labels[index] = parts[i];
                                components.add(parts[i]);
                            } else {
                                dataset[index][i] = Double.parseDouble(parts[i]);
                            }
                        }
                        index++;
                    }
                }
                line = br.readLine();
            }
        }

        Collections.shuffle(indices);

        double[][] randomDataset = new double[lines][dim];
        String[] randomLabels = new String[lines];
        for (int i = 0; i < indices.size(); i++) {
            randomDataset[i] = dataset[indices.get(i)];
            randomLabels[i] = labels[indices.get(i)];
        }

        dataset = randomDataset;
        labels = randomLabels;

        int start = 0;
        int end = foldSize;
        for (int i = 0; i < FOLDS; i++) {
            int indexTest = 0;
            int indexTraining = 0;
            for (int j = 0; j < lines; j++) {
                if (j >= start && j < end) {
                    testset[i][indexTest++] = dataset[j];
                } else{
                    trainingset[i][indexTraining++] = dataset[j];
                }
            }

            start += foldSize;
            end += foldSize;
        }

        if (groundTruthDistribution && means != null && covariances != null && weights != null) {
            trueDistribution = new MixtureMultivariateNormalDistribution(weights, means, covariances);
        }
    }

    // Implementing Fisher–Yates shuffle
    private void shuffleArray(int[] ar)
    {
        // If running on Java 6 or older, use `new Random()` on RHS here
        Random rnd = ThreadLocalRandom.current();
        for (int i = ar.length - 1; i > 0; i--)
        {
            int index = rnd.nextInt(i + 1);
            // Simple swap
            int a = ar[index];
            ar[index] = ar[i];
            ar[i] = a;
        }
    }


    public double[][][] getTrainingset() {
        return trainingset;
    }

    public double[][][] getTestset() {
        return testset;
    }

    public MixtureMultivariateNormalDistribution getTrueDistribution() {
        return trueDistribution;
    }

    public double[][] getDataset() {
        return dataset;
    }

    public String[] getLabels() {
        return labels;
    }
}
