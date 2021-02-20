import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner;

import java.io.IOException;
import java.util.List;

public class PMI {
    // Pointwise Mutual Information, PMI

    private final static String concatFlagForWord = GlobalSetting.concatFlagForWord;
    private final static String concatFlagForPair = GlobalSetting.concatFlagForPair;
    private final static String punctuationFlag = GlobalSetting.punctuationFlag;
    private final static String tempPath = GlobalSetting.tempPath;

    private static int sumWord = 0;

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();

        Job tempJob = Job.getInstance(conf, "preWordSegmentEntropyJob");
        tempJob.setJarByClass(PMI.class);
        tempJob.setMapperClass(PrePlusTokenizerMapper.class);
        tempJob.setCombinerClass(SumCombiner.class);
        tempJob.setPartitionerClass(ConcatWordsSamePartitioner.class);
        tempJob.setReducerClass(DivReducer.class);
        tempJob.setMapOutputKeyClass(Text.class);
        tempJob.setMapOutputValueClass(DoubleWritable.class);
        tempJob.setOutputKeyClass(Text.class);
        tempJob.setOutputValueClass(DoubleWritable.class);
        FileInputFormat.addInputPath(tempJob, new Path(args[0]));
        FileOutputFormat.setOutputPath(tempJob, new Path(tempPath));

        Job PMIJob = Job.getInstance(conf, "PMIJob");
        PMIJob.setJarByClass(PMI.class);
        PMIJob.setMapperClass(NormalMapper.class);
        PMIJob.setPartitionerClass(ConcatWordsSamePartitioner.class);
        PMIJob.setReducerClass(PMIReducer.class);
        PMIJob.setOutputKeyClass(Text.class);
        PMIJob.setOutputValueClass(DoubleWritable.class);
        FileInputFormat.addInputPath(PMIJob, new Path(tempPath + "/part-r-00000"));
        FileOutputFormat.setOutputPath(PMIJob, new Path(args[1]));

        if (tempJob.waitForCompletion(true)) {
            System.exit(PMIJob.waitForCompletion(true) ? 0 : 1);
        }
    }

    public static class PrePlusTokenizerMapper
            extends Mapper<Object, Text, Text, DoubleWritable> {
        // Map: (Object key, Text line) -> emit: (<ConcatWords, preWord >, 1)
        private final static DoubleWritable one = new DoubleWritable(1);
        private Text word = new Text();// 输出的key类型

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {
            List<String> splitText = Utils.splitLine(value);

            for (String splitWord : splitText) {
                if (punctuationFlag.equals(splitWord)) {
                    continue;
                }
                word.set(splitWord);
                context.write(word, one);
                PMI.sumWord += 1;
            }

            for (int i = 1; i + 1 < splitText.size(); i++) {
                String wordPairLeft = splitText.get(i);
                String wordPairRight = splitText.get(i + 1);

                if (punctuationFlag.equals(wordPairLeft) || punctuationFlag.equals(wordPairRight)) {
                    continue;
                }
                // emit: (<ConcatWords>, 1)
                word.set(wordPairLeft + concatFlagForWord + wordPairRight);
                context.write(word, one);
            }
        }
    }

    public static class NormalMapper
            extends Mapper<Object, Text, Text, DoubleWritable> {
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] list = value.toString().split("\t");
            String word = list[0];
            double val = Double.parseDouble(list[1]);
            context.write(new Text(word), new DoubleWritable(val));
        }
    }

    public static class SumCombiner
            extends Reducer<Text, DoubleWritable, Text, DoubleWritable> {
        // Combine: (Text key, IntWritable value) -> emit: (key, sum)
        private DoubleWritable result = new DoubleWritable();

        public void reduce(Text key, Iterable<DoubleWritable> values,
                           Context context
        ) throws IOException, InterruptedException {
            double sum = 0;
            for (DoubleWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result);
        }
    }

    public static class ConcatWordsSamePartitioner
            extends HashPartitioner<Text, DoubleWritable> {
        @Override
        public int getPartition(Text key, DoubleWritable value, int numReduceTasks) {
            String concatWords = key.toString().split(concatFlagForPair)[0];
            return super.getPartition(new Text(concatWords), value, numReduceTasks);
        }
    }

    public static class DivReducer
            extends Reducer<Text, DoubleWritable, Text, DoubleWritable> {
        // Combine: (Text key, IntWritable value) -> emit: (key, sum)
        private String pastConcatWords = null;                     // Record the Concat Words
        private double numConcatWords = 0;

        public void reduce(Text key, Iterable<DoubleWritable> values, Context context) throws IOException, InterruptedException {
            String[] stringLists = key.toString().split(concatFlagForWord);
            String concatWords = stringLists[0];
            double sum = 0;
            for (DoubleWritable item : values) {
                sum += item.get();
            }
            if (stringLists.length == 1) {
                pastConcatWords = concatWords;
                numConcatWords = sum;
            } else if (concatWords.equals(pastConcatWords)) {
                key = new Text(stringLists[1] + concatFlagForWord + stringLists[0]);
                sum /= numConcatWords;
            } else {
                System.exit(1);
            }
            context.write(key, new DoubleWritable(sum));
        }
    }

    public static class PMIReducer
            extends Reducer<Text, DoubleWritable, Text, DoubleWritable> {
        private String pastConcatWords = null;                     // Record the Concat Words
        private double numConcatWords = 0;

        public void reduce(Text key, Iterable<DoubleWritable> values, Context context) throws IOException, InterruptedException {
            String[] stringLists = key.toString().split(concatFlagForWord);
            String concatWords = stringLists[0];
            double sum = 0;
            for (DoubleWritable item : values) {
                sum += item.get();
            }
            if (stringLists.length == 1) {
                pastConcatWords = concatWords;
                numConcatWords = sum;
            } else if (concatWords.equals(pastConcatWords)) {
                key = new Text(stringLists[1] + concatFlagForWord + stringLists[0]);
                sum /= numConcatWords;
                double pmiScore = Math.log(sum * PMI.sumWord);
                context.write(key, new DoubleWritable(pmiScore));
            } else {
                System.exit(1);
            }
        }
    }
}
