import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputCommitter;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.wltea.analyzer.lucene.IKAnalyzer;

import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;


public class Entropy {
    private final static String concatFlagForWord = "&";
    private final static String concatFlagForPair = "#";
    private final static String punctuationFlag = "符号";
    private final static int filterFrequencyLimit = 10;


    public static void main(String[] args) throws Exception {
        // preWordEntropyJob
        Configuration conf = new Configuration();
        Job preWordEntropyJob = Job.getInstance(conf, "preWordEntropyJob");
        preWordEntropyJob.setJarByClass(Entropy.class);
        preWordEntropyJob.setMapperClass(PreTokenizerMapper.class);
        preWordEntropyJob.setCombinerClass(SumCombiner.class);
        preWordEntropyJob.setPartitionerClass(ConcatWordsSamePartitioner.class);
        preWordEntropyJob.setReducerClass(FrequencyCombinationReducer.class);
        preWordEntropyJob.setOutputKeyClass(Text.class);
        preWordEntropyJob.setOutputValueClass(IntWritable.class);
        preWordEntropyJob.setOutputFormatClass(PreWordEntropyOutputFormat.class);
        FileInputFormat.addInputPath(preWordEntropyJob, new Path(args[0]));
        FileOutputFormat.setOutputPath(preWordEntropyJob, new Path(args[1]));

        // postWordEntropyJob
        Job postWordEntropyJob = Job.getInstance(new Configuration(), "preWordEntropyJob");
        postWordEntropyJob.setJarByClass(Entropy.class);
        postWordEntropyJob.setMapperClass(PostTokenizerMapper.class);
        postWordEntropyJob.setCombinerClass(SumCombiner.class);
        postWordEntropyJob.setPartitionerClass(ConcatWordsSamePartitioner.class);
        postWordEntropyJob.setReducerClass(FrequencyCombinationReducer.class);
        postWordEntropyJob.setOutputKeyClass(Text.class);
        postWordEntropyJob.setOutputValueClass(IntWritable.class);
        postWordEntropyJob.setOutputFormatClass(PostWordEntropyOutputFormat.class);
        FileInputFormat.addInputPath(postWordEntropyJob, new Path(args[0]));
        FileOutputFormat.setOutputPath(postWordEntropyJob, new Path(args[1]));

        preWordEntropyJob.submit();
        postWordEntropyJob.submit();

        boolean flag = preWordEntropyJob.isComplete() & postWordEntropyJob.isComplete();
        while (!flag) {
            flag = preWordEntropyJob.isComplete() & postWordEntropyJob.isComplete();
        }
        System.exit(0);
    }


    private static List<String> getAnalyzedStr(Analyzer analyzer, String content) throws Exception {
        TokenStream stream = analyzer.tokenStream(null, new StringReader(content));
        CharTermAttribute term = stream.addAttribute(CharTermAttribute.class);

        List<String> result = new ArrayList<>();
        while (stream.incrementToken()) {
            result.add(term.toString());
        }

        return result;
    }

    private static boolean hasPunctuation(String content) {
        return content.length() != content.replaceAll("[\\pP‘’“”]", "").length();
    }

    private static String replacePunctuationWithFlags(String content) {
        return punctuationFlag + content.replaceAll("[\\pP‘’“”]", punctuationFlag) + punctuationFlag;
    }

    private static List<String> splitLine(Text value) {
        String line = replacePunctuationWithFlags(value.toString());
        Analyzer analyzer = new IKAnalyzer();
        List<String> splitText = null;
        try {
            splitText = getAnalyzedStr(analyzer, line);
        } catch (Exception e) {
            e.printStackTrace();
        }
        assert splitText != null;

        return splitText;
    }


    public static class PreTokenizerMapper
            extends Mapper<Object, Text, Text, IntWritable> {
        // Map: (Object key, Text line) -> emit: (<ConcatWords, preWord >, 1)
        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();// 输出的key类型

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {
            List<String> splitText = splitLine(value);

            for (int i = 1; i + 1 < splitText.size(); i++) {
                String wordPairLeft = splitText.get(i);
                String wordPairRight = splitText.get(i + 1);

                if (punctuationFlag.equals(wordPairLeft) || punctuationFlag.equals(wordPairRight)) {
                    continue;
                }
                // emit: (<ConcatWords, preWord>, 1)
                String concatWords = wordPairLeft + concatFlagForWord + wordPairRight;
                word.set(concatWords + concatFlagForPair + splitText.get(i - 1));
                context.write(word, one);
            }
        }
    }

    public static class PostTokenizerMapper
            extends Mapper<Object, Text, Text, IntWritable> {
        // Map: (Object key, Text line) -> emit: (<ConcatWords, preWord >, 1)
        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();// 输出的key类型

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {

            List<String> splitText = splitLine(value);
            for (int i = 0; i < splitText.size() - 2; i++) {
                String wordPairLeft = splitText.get(i);
                String wordPairRight = splitText.get(i + 1);

                if (punctuationFlag.equals(wordPairLeft) || punctuationFlag.equals(wordPairRight)) {
                    continue;
                }

                // emit: (<ConcatWords, postWord>, 1)
                String concatWords = wordPairLeft + concatFlagForWord + wordPairRight;
                word.set(concatWords + concatFlagForPair + splitText.get(i + 2));
                context.write(word, one);
            }
        }
    }


    public static class SumCombiner
            extends Reducer<Text, IntWritable, Text, IntWritable> {
        // Combine: (Text key, IntWritable value) -> emit: (key, sum)
        private IntWritable result = new IntWritable();

        public void reduce(Text key, Iterable<IntWritable> values,
                           Context context
        ) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result);
        }
    }


    public static class ConcatWordsSamePartitioner
            extends HashPartitioner<Text, IntWritable> {
        @Override
        public int getPartition(Text key, IntWritable value, int numReduceTasks) {
            String concatWords = key.toString().split(concatFlagForPair)[0];
            return super.getPartition(new Text(concatWords), value, numReduceTasks);
        }
    }


    public static class FrequencyCombinationReducer
            extends Reducer<Text, IntWritable, Text, Text> {
        // Combine: (Text key, IntWritable value) -> emit: (key, sum)
        private String pastConcatWords = null;                     // Record the Concat Words
        private String emitOutput = "";
        private int sumConcatWords = 0;

        public void reduce(Text key, Iterable<IntWritable> values,
                           Context context
        ) throws IOException, InterruptedException {
            String[] stringLists = key.toString().split(concatFlagForPair);
            String concatWords = stringLists[0];
            String neighbourWords = stringLists[1];
            if (null == pastConcatWords) {
                pastConcatWords = concatWords;
            }

            if (!concatWords.equals(pastConcatWords)) {
                if (sumConcatWords >= filterFrequencyLimit) {
                    emitOutput = ":" + sumConcatWords + ";" + emitOutput;
                    context.write(new Text(pastConcatWords), new Text(emitOutput));
                }
                // clear
                pastConcatWords = concatWords;
                emitOutput = "";
                sumConcatWords = 0;
            }

            int count = 0;
            for (IntWritable val : values) {
                count += val.get();
            }
            emitOutput += neighbourWords + ":" + count + ";";
            sumConcatWords += count;
        }

        public void cleanup(Context context) throws IOException, InterruptedException {
            if (pastConcatWords != null && sumConcatWords >= filterFrequencyLimit) {
                emitOutput = ":" + sumConcatWords + ";" + emitOutput;
                context.write(new Text(pastConcatWords), new Text(emitOutput));
            }
            super.cleanup(context);
        }
    }

    public static class PreWordEntropyOutputFormat extends TextOutputFormat {
        @Override
        public Path getDefaultWorkFile(TaskAttemptContext context, String extension) throws IOException {
            FileOutputCommitter committer = (FileOutputCommitter) getOutputCommitter(context);
            return new Path(committer.getWorkPath(), "PreWordEntropy.txt");
        }
    }

    public static class PostWordEntropyOutputFormat extends TextOutputFormat {
        @Override
        public Path getDefaultWorkFile(TaskAttemptContext context, String extension) throws IOException {
            FileOutputCommitter committer = (FileOutputCommitter) getOutputCommitter(context);
            return new Path(committer.getWorkPath(), "PostWordEntropy.txt");//getOutputName(context)
        }
    }
}
