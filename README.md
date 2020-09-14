# HCEM
Model-based clustering algorithm based on HDBSCAN* hiearchies.

Based on our ECML-PKDD 2020 paper: "Model-based Clustering with HDBSCAN*"

Maven is required to run the code.
 
mvn clean install <br>
mvn exec:java -D exec.mainClass=ca.ualberta.cs.HCEM -Dexec.args=&quot;&lt;args&gt;&quot;

Required arguments with parameters: <br>
-f --filename &lt;path to file&gt;
-c --criterion BIC, ICL or VAL

Optional flags: <br>
-l --labels if labels are available for ARI calculation
-d --distribution if original distribution is available for KLD calculation

See test_dataset.dat for data format: Space as separator, labels at the end of each line and distribution parameters at the end of file.