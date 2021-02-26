import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.List;

public class PMI {
    // Pointwise Mutual Information, PMI

    private final static String tempPath = GlobalSetting.tempPath;

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        conf.set("concatFlagForWord", GlobalSetting.concatFlagForWord);
        conf.set("concatFlagForPair", GlobalSetting.concatFlagForPair);
        conf.set("punctuationFlag", GlobalSetting.concatFlagForWord);
        conf.set("filterFrequencyLimit", String.valueOf(GlobalSetting.filterFrequencyLimit));
        conf.set("sumFlag", GlobalSetting.sumFlag);

        FileSystem fileSystem = FileSystem.get(new URI(args[0]), conf);
        Utils.setFileSystem(fileSystem);

        Job tempJob = Job.getInstance(conf, "tempJob");
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
        fileSystem.delete(new Path(GlobalSetting.tempPath), true);
        tempJob.waitForCompletion(true);

        double sumWord = getSumWord(fileSystem, new Path(tempPath + "/part-r-00000"));
        conf.set("sumWord", String.valueOf(sumWord));

        Job PMIJob = Job.getInstance(conf, "PMIJob");
        PMIJob.setJarByClass(PMI.class);
        PMIJob.setMapperClass(NormalMapper.class);
        PMIJob.setPartitionerClass(ConcatWordsSamePartitioner.class);
        PMIJob.setReducerClass(PMIReducer.class);
        PMIJob.setOutputKeyClass(Text.class);
        PMIJob.setOutputValueClass(DoubleWritable.class);
        FileInputFormat.addInputPath(PMIJob, new Path(tempPath + "/part-r-00000"));
        FileOutputFormat.setOutputPath(PMIJob, new Path(args[1]));

        fileSystem.delete(new Path(GlobalSetting.PMIPath), true);
        fileSystem.delete(new Path(args[1]), true);

        PMIJob.waitForCompletion(true);
    }

    static double getSumWord(FileSystem fileSystem, Path path) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(fileSystem.open(path)));
        String line;
        while ((line = reader.readLine()) != null) {
            String[] stringLists = line.split("\t");
            if (stringLists[0].equals(GlobalSetting.sumFlag)) {
                return Double.parseDouble(stringLists[1]);
            }
        }
        throw new NullPointerException("Didn't know sum word!");
    }

    public static class PrePlusTokenizerMapper
            extends Mapper<Object, Text, Text, DoubleWritable> {
        // Map: (Object key, Text line) -> emit: (<ConcatWords, preWord >, 1)
        private final static DoubleWritable one = new DoubleWritable(1);
        private Text word = new Text();// 输出的key类型

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {
            String punctuationFlag = context.getConfiguration().get("punctuationFlag");
            String concatFlagForWord = context.getConfiguration().get("concatFlagForWord");
            String sumFlag = context.getConfiguration().get("sumFlag");

            List<String> splitText = Utils.splitLine(value);

            for (String splitWord : splitText) {
                if (punctuationFlag.equals(splitWord)) {
                    continue;
                }
                word.set(splitWord);
                context.write(word, one);
            }

            context.write(new Text(sumFlag), new DoubleWritable(splitText.size()));

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
            System.out.println(GlobalSetting.concatFlagForPair); // Test which jvm
            String concatWords = key.toString().split(GlobalSetting.concatFlagForPair)[0];
            return super.getPartition(new Text(concatWords), value, numReduceTasks);
        }
    }

    public static class DivReducer
            extends Reducer<Text, DoubleWritable, Text, DoubleWritable> {
        // Combine: (Text key, IntWritable value) -> emit: (key, sum)
        private String pastConcatWords = null;                     // Record the Concat Words
        private double numConcatWords = 0;

        public void reduce(Text key, Iterable<DoubleWritable> values, Context context) throws IOException, InterruptedException {
            String concatFlagForWord = context.getConfiguration().get("concatFlagForWord");

            String[] stringLists = key.toString().split(concatFlagForWord);
            double sum = 0;
            for (DoubleWritable item : values) {
                sum += item.get();
            }
            String concatWords = stringLists[0];

            if (stringLists.length == 1) {
                pastConcatWords = concatWords;
                numConcatWords = sum;
            } else if (stringLists.length == 2 && concatWords.equals(pastConcatWords)) {
                key = new Text(stringLists[1] + concatFlagForWord + stringLists[0]);
                sum /= numConcatWords;
            } else {
                System.out.println(key);
//                return ;
            }
            context.write(key, new DoubleWritable(sum));
        }
    }

    public static class PMIReducer
            extends Reducer<Text, DoubleWritable, Text, DoubleWritable> {
        private String pastConcatWords = null;                     // Record the Concat Words
        private double numConcatWords = 0;
        private double sumWord = 1e7;


        public void setup(Context context)
                throws IOException, InterruptedException, NullPointerException {
            super.setup(context);
            sumWord = Double.parseDouble(context.getConfiguration().get("sumWord"));
        }

        public void reduce(Text key, Iterable<DoubleWritable> values, Context context) throws IOException, InterruptedException {
            String concatFlagForWord = context.getConfiguration().get("concatFlagForWord");

            String[] stringLists = key.toString().split(concatFlagForWord);
            String concatWords = stringLists[0];
            double sum = 0;
            for (DoubleWritable item : values) {
                sum += item.get();
            }
            if (stringLists.length == 1) {
                pastConcatWords = concatWords;
                numConcatWords = sum;
            } else if (stringLists.length == 2 && concatWords.equals(pastConcatWords)) {
                key = new Text(stringLists[1] + concatFlagForWord + stringLists[0]);
                double pmiScore = Math.log(sum * sumWord / numConcatWords);
                context.write(key, new DoubleWritable(pmiScore));
            } else {
//                System.exit(1);
                System.out.println(key);
            }
        }
    }
}
