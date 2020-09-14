package ca.ualberta.cs;

import ca.ualberta.cs.distance.DistanceCalculator;
import ca.ualberta.cs.distance.EuclideanDistance;
import ca.ualberta.cs.hdbscanstar.Cluster;
import ca.ualberta.cs.hdbscanstar.HDBSCANStar;
import ca.ualberta.cs.hdbscanstar.UndirectedGraph;

import java.io.IOException;
import java.util.ArrayList;

/**
 *
 * @author Michael Strobl
 */
public class HDBSCANRunner {

    public static String startHDBSCAN(String criterion, String inputFile, double[][] dataSet, int minPoints) throws IOException {

        HDBSCANStarParameters parameters = checkInputParameters(criterion, inputFile,minPoints);
        int numPoints = dataSet.length;

        long startTime = System.currentTimeMillis();
        double[] coreDistances = HDBSCANStar.calculateCoreDistances(dataSet, parameters.minPoints, parameters.distanceFunction);

        UndirectedGraph mst = HDBSCANStar.constructMST(dataSet, coreDistances, true, parameters.distanceFunction);
        mst.quicksortByEdgeWeight();

        dataSet = null;

        double[] pointNoiseLevels = new double[numPoints];
        int[] pointLastClusters = new int[numPoints];

        ArrayList<Cluster> clusters = null;
        try {
            startTime = System.currentTimeMillis();
            clusters = HDBSCANStar.computeHierarchyAndClusterTree(mst, parameters.minClusterSize,
                    parameters.compactHierarchy, null, parameters.hierarchyFile,
                    null, ",", pointNoiseLevels, pointLastClusters, null);
        }
        catch (IOException ioe) {
            System.err.println("Error writing to hierarchy file or cluster tree file.");
            System.exit(-1);
        }

        return parameters.hierarchyFile;
    }

    private static HDBSCANStarParameters checkInputParameters(String criterion, String inputFile, int minPoints) {
        HDBSCANStarParameters parameters = new HDBSCANStarParameters();
        parameters.distanceFunction = new EuclideanDistance();
        parameters.compactHierarchy = true;
        parameters.inputFile = inputFile;
        String inputName = parameters.inputFile;
        if (parameters.inputFile.contains("."))
            inputName = parameters.inputFile.substring(0, parameters.inputFile.lastIndexOf("."));

        parameters.hierarchyFile = inputName + "_" + criterion + "_hierarchy.csv";
        parameters.minPoints = minPoints;
        parameters.minClusterSize = minPoints;
        parameters.distanceFunction = new EuclideanDistance();
        return parameters;
    }

    /**
     * Simple class for storing input parameters.
     */
    private static class HDBSCANStarParameters {
        public String inputFile;
        public String constraintsFile;
        public Integer minPoints;
        public Integer minClusterSize;
        public boolean compactHierarchy;
        public DistanceCalculator distanceFunction;

        public String hierarchyFile;
    }
}
