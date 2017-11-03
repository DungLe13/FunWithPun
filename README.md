# FUN WITH PUN

Pun is a form of wordplay that exploits the different meanings of the same word (homonyms) or the fact that there are words that sound alike but have different meanings (heteronyms). From Shakespeare’s plays to daily life conversations, pun has been used widely as a way to express humor. The two types of pun are expressed in the following sentences:

> I used to be a banker but I lost interest
>
> The dentist had a bad day at the orifice

In the first instance, the word 'interest' can be understood in two ways: i. the state of wanting to know or learn about sth, or ii. the money paid regularly at a particular rate. In the latter example, the word 'orifice' sounds similar to the word 'office', and therefore can be punned as heteronyms.

## Three subtasks
The tasks described below are rephrased from the tasks described in *SemEval-2017 Task 7: Detection and Interpretation of English Puns*

**Task 1:** *(Pun Detection)* Given a document of sentences that either contain a pun word or not, classify sentences with pun word (1) from those without pun word (0)

**Task 2:** *(Pun Location)* For each context that has pun word, locate the pun word (or which word is the pun word)

**Task 3:** *(Pun Interpretation)* For each context that has pun word, the system need to annotate two different senses of the pun word in WordNet

## Scorer
***Disclaimer:*** The content in this section is taken from the README.md file that can be downloaded from https://www.ukp.tu-darmstadt.de/data/sense-labelling-resources/sense-annotated-english-puns/

The `scorer/src` directory contains the Java source code for the
scoring software and the `scorer/bin` directory contains the compiled
classes.

To (re)compile the scoring software:

```
$ cd scorer
$ javac -d bin src/de/tudarmstadt/ukp/semeval2017/task7/scorer/PunScorer.java
```

To run the scoring software:

```
$ cd scorer/bin
$ java de.tudarmstadt.ukp.semeval2017.task7.scorer.PunScorer
```

Running the scorer without any command-line arguments prints the
following usage instructions:

```
Usage:
        java de.tudarmstadt.ukp.semeval2017.task7.scorer.PunScorer [ -d | -l | -i ] <goldFile> <resultFile> [ <outputFile> ]
```

The first command-line argument is required, and must be either `-d`
(for the detection subtask), `-l` (for the location subtask), or `-i`
(for the interpretation subtask).  The next two arguments are also
required; they specify the location of the gold-standard file and the
system result file, in that order.  The final optional argument
specifies the location of a file to write the output.  If the argument
is omitted, output is written to standard output.