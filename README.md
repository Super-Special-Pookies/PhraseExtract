# PhraseExtract

This project is used for distributed phrase extract based on a mature tokenizer: IkAnalyzer. 
We mainly use Map-Reduce to construct the pipeline, a version of spark is on the way.


## Dataset

We use FinancialDatasets supported by https://github.com/smoothnlp/FinancialDatasets

## Main Pipeline

We calculate three kinds of score: PMI(Pointwise Mutual Information), Entropy and Segment Entropy.

The final score = PMI + Entropy - Segment Entropy.

We calculate these successively and sort the final results.

## How to use 

We use maven to manage dependencies. The version in detail can be found in pom.xml. 

A mature tokenizer needs to be installed in your local Maven Repository. 
A easy tutorial is supported here: https://github.com/wks/ik-analyzer .

You can replace with a DIY tokenizer to split the sentence as you want in `Utils.splitLine`.

Every Class has a main to test whether its function work well. 
The `Main.main` fuse all and you can pass 2 parameters: InputPath and OutputPath to run.

## FAQ

1. Connection Timed Out while running preWordEntropyJob and postWordEntropyJob:

    > try to start the history server: 
    > 
    > ```bash
    > bash mr-jobhistory-daemon.sh start historyserver
    > ```

2. Hadoop java.io.IOException: Mkdirs failed to create /some/path:

    > Remove META-INF/LICENSE as [here](https://stackoverflow.com/questions/10522835/hadoop-java-io-ioexception-mkdirs-failed-to-create-some-path) talked
    >
    > An example: 
    > ```bash
    > zip -d out/artifacts/PhraseExtract_jar/PhraseExtract.jar META-INF/LICENSE
    > zip -d out/artifacts/PhraseExtract_jar/PhraseExtract.jar LICENSE
    > ```

