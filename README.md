# Guide to Modifying code for Various Modelling Methods
Regardless of attributes used to modeling, the overall flow should be the same.
Overall process should be: read in data, process into modeling format, split into training and testing, convert to LabeledPoint object, pass to SVM.

## Dependencies
- Apache Spark
- NumPy(for Spark's SVM to run)
- YARN(optional)

## Other Notes
- Make sure all the CSV files are in the same directory on HDFS
- Changes the code that reads from CSV files to point to the correct directory on HDFS
- Every code chunk has a different level of data granularity. Edit the code depending on what you feel is the correct granularity/set
