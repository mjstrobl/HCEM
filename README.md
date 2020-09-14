# HCEM

##Description

Based on our ECML-PKDD 2020 paper: "Model-based Clustering with HDBSCAN*"

##Dependencies

Download the HDBSCAN implementation:

http://lapad-web.icmc.usp.br/?portfolio_1=a-handful-of-experiments

Copy &quot;HDBSCAN_Star.jar&quot; to the repository.

Maven is required to run the code. Run the following maven commands:
1. mvn install:install-file -Dfile=&lt;path to HDBSCAN_Star.jar&gt; -DgroupId=ca.ualberta.cs -DartifactId=HDBSCAN -Dversion=1.0 -Dpackaging=jar -DgeneratePom=true
2. mvn clean install <br>

##Run

Run the program with the following command, description of flags below:

mvn exec:java -D exec.mainClass=ca.ualberta.cs.HCEM -Dexec.args=&quot;&lt;args&gt;&quot;

Required arguments with parameters: <br>
-f --filename &lt;path to file&gt;
-c --criterion BIC, ICL or VAL

Optional flags: <br>
-l --labels if labels are available for ARI calculation
-d --distribution if original distribution is available for KLD calculation

See test_dataset.dat for data format: Space as separator, labels at the end of each line and distribution parameters at the end of file.