import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
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

import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.List;


public class Entropy {
    public static void main(String[] args) throws Exception {

        ArrayList<String> outputPathsList = generateOutputPathsList(args[1], 4);
        Configuration conf = new Configuration();
        conf.set("concatFlagForWord", GlobalSetting.concatFlagForWord);
        conf.set("concatFlagForPair", GlobalSetting.concatFlagForPair);
        conf.set("punctuationFlag", GlobalSetting.concatFlagForWord);
        conf.set("filterFrequencyLimit", String.valueOf(GlobalSetting.filterFrequencyLimit));
        FileSystem fileSystem = FileSystem.get(new URI(args[0]), conf);


        // preWordEntropyJob
        Job preWordEntropyJob = Job.getInstance(conf, "preWordEntropyJob");
        preWordEntropyJob.setJarByClass(Entropy.class);
        preWordEntropyJob.setMapperClass(PreTokenizerMapper.class);
        preWordEntropyJob.setCombinerClass(SumCombiner.class);
        preWordEntropyJob.setPartitionerClass(ConcatWordsSamePartitioner.class);
        preWordEntropyJob.setReducerClass(FrequencyCombinationReducer.class);
        preWordEntropyJob.setMapOutputKeyClass(Text.class);
        preWordEntropyJob.setMapOutputValueClass(IntWritable.class);
        preWordEntropyJob.setOutputKeyClass(Text.class);
        preWordEntropyJob.setOutputValueClass(Text.class);
        preWordEntropyJob.setOutputFormatClass(PreWordEntropyOutputFormat.class);
        FileInputFormat.addInputPath(preWordEntropyJob, new Path(args[0]));
        FileOutputFormat.setOutputPath(preWordEntropyJob, new Path(outputPathsList.get(0)));

        // postWordEntropyJob
        Job postWordEntropyJob = Job.getInstance(conf, "postWordEntropyJob");
        postWordEntropyJob.setJarByClass(Entropy.class);
        postWordEntropyJob.setMapperClass(PostTokenizerMapper.class);
        postWordEntropyJob.setCombinerClass(SumCombiner.class);
        postWordEntropyJob.setPartitionerClass(ConcatWordsSamePartitioner.class);
        postWordEntropyJob.setReducerClass(FrequencyCombinationReducer.class);
        postWordEntropyJob.setMapOutputKeyClass(Text.class);
        postWordEntropyJob.setMapOutputValueClass(IntWritable.class);
        postWordEntropyJob.setOutputKeyClass(Text.class);
        postWordEntropyJob.setOutputValueClass(Text.class);
        postWordEntropyJob.setOutputFormatClass(PostWordEntropyOutputFormat.class);
        FileInputFormat.addInputPath(postWordEntropyJob, new Path(args[0]));
        FileOutputFormat.setOutputPath(postWordEntropyJob, new Path(outputPathsList.get(1)));

        // preWordSegmentEntropyJob
        Job preWordSegmentEntropyJob = Job.getInstance(conf, "preWordSegmentEntropyJob");
        preWordSegmentEntropyJob.setJarByClass(Entropy.class);
        preWordSegmentEntropyJob.setMapperClass(PreSegmentTokenizerMapper.class);
        preWordSegmentEntropyJob.setCombinerClass(SumCombiner.class);
        preWordSegmentEntropyJob.setPartitionerClass(ConcatWordsSamePartitioner.class);
        preWordSegmentEntropyJob.setReducerClass(FrequencyCombinationReducer.class);
        preWordSegmentEntropyJob.setMapOutputKeyClass(Text.class);
        preWordSegmentEntropyJob.setMapOutputValueClass(IntWritable.class);
        preWordSegmentEntropyJob.setOutputKeyClass(Text.class);
        preWordSegmentEntropyJob.setOutputValueClass(Text.class);
        preWordSegmentEntropyJob.setOutputFormatClass(PreWordSegmentEntropyOutputFormat.class);
        FileInputFormat.addInputPath(preWordSegmentEntropyJob, new Path(args[0]));
        FileOutputFormat.setOutputPath(preWordSegmentEntropyJob, new Path(outputPathsList.get(2)));

        // postWordSegmentEntropyJob
        Job postWordSegmentEntropyJob = Job.getInstance(conf, "postWordSegmentEntropyJob");
        postWordSegmentEntropyJob.setJarByClass(Entropy.class);
        postWordSegmentEntropyJob.setMapperClass(PostSegmentTokenizerMapper.class);
        postWordSegmentEntropyJob.setCombinerClass(SumCombiner.class);
        postWordSegmentEntropyJob.setPartitionerClass(ConcatWordsSamePartitioner.class);
        postWordSegmentEntropyJob.setReducerClass(FrequencyCombinationReducer.class);
        postWordSegmentEntropyJob.setMapOutputKeyClass(Text.class);
        postWordSegmentEntropyJob.setMapOutputValueClass(IntWritable.class);
        postWordSegmentEntropyJob.setOutputKeyClass(Text.class);
        postWordSegmentEntropyJob.setOutputValueClass(Text.class);
        postWordSegmentEntropyJob.setOutputFormatClass(PostWordSegmentEntropyOutputFormat.class);
        FileInputFormat.addInputPath(postWordSegmentEntropyJob, new Path(args[0]));
        FileOutputFormat.setOutputPath(postWordSegmentEntropyJob, new Path(outputPathsList.get(3)));

        for (int i = 0; i < 4; i++) {
            fileSystem.delete(new Path(outputPathsList.get(i)), true);
        }
        preWordSegmentEntropyJob.submit();
        postWordSegmentEntropyJob.submit();
        preWordEntropyJob.submit();
        postWordEntropyJob.submit();


        boolean flag = preWordEntropyJob.isComplete() & postWordEntropyJob.isComplete() &
                preWordSegmentEntropyJob.isComplete() & postWordSegmentEntropyJob.isComplete();
        while (!flag) {
            flag = preWordEntropyJob.isComplete() & postWordEntropyJob.isComplete() &
                    preWordSegmentEntropyJob.isComplete() & postWordSegmentEntropyJob.isComplete();
        }

        System.exit(0);
    }

    private static ArrayList<String> generateOutputPathsList(String rootPath, int pathNum) {
        ArrayList<String> results = new ArrayList<>();
        for (int i = 0; i < pathNum; i++) {
            results.add(rootPath + i);
        }
        return results;
    }


    public static class PreTokenizerMapper
            extends Mapper<Object, Text, Text, IntWritable> {
        // Map: (Object key, Text line) -> emit: (<ConcatWords, preWord >, 1)
        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();// 输出的key类型

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {
            String concatFlagForWord = context.getConfiguration().get("concatFlagForWord");
            String concatFlagForPair = context.getConfiguration().get("concatFlagForPair");
            String punctuationFlag = context.getConfiguration().get("punctuationFlag");

            List<String> splitText = Utils.splitLine(value);

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
            String concatFlagForWord = context.getConfiguration().get("concatFlagForWord");
            String concatFlagForPair = context.getConfiguration().get("concatFlagForPair");
            String punctuationFlag = context.getConfiguration().get("punctuationFlag");

            List<String> splitText = Utils.splitLine(value);
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

    public static class PreSegmentTokenizerMapper
            extends Mapper<Object, Text, Text, IntWritable> {
        // Map: (Object key, Text line) -> emit: (<Segment, preWord >, 1)
        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();// 输出的key类型

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {
            String concatFlagForPair = context.getConfiguration().get("concatFlagForPair");
            String punctuationFlag = context.getConfiguration().get("punctuationFlag");

            List<String> splitText = Utils.splitLine(value);

            for (int i = 1; i < splitText.size(); i++) {
                String wordPairLeft = splitText.get(i);

                if (punctuationFlag.equals(wordPairLeft)) {
                    continue;
                }
                // emit: (<Segment, preWord>, 1)
                word.set(wordPairLeft + concatFlagForPair + splitText.get(i - 1));
                context.write(word, one);
            }
        }
    }


    public static class PostSegmentTokenizerMapper
            extends Mapper<Object, Text, Text, IntWritable> {
        // Map: (Object key, Text line) -> emit: (<Segment, postWord >, 1)
        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();// 输出的key类型

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {
            String concatFlagForPair = context.getConfiguration().get("concatFlagForPair");
            String punctuationFlag = context.getConfiguration().get("punctuationFlag");
            List<String> splitText = Utils.splitLine(value);

            for (int i = 0; i + 1 < splitText.size(); i++) {
                String wordPairRight = splitText.get(i);

                if (punctuationFlag.equals(wordPairRight)) {
                    continue;
                }
                // emit: (<Segment, postWord>, 1)
                word.set(wordPairRight + concatFlagForPair + splitText.get(i + 1));
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
            String concatWords = key.toString().split(GlobalSetting.concatFlagForPair)[0];
            return super.getPartition(new Text(concatWords), value, numReduceTasks);
        }
    }


    public static class FrequencyCombinationReducer
            extends Reducer<Text, IntWritable, Text, Text> {
        // Reducer: (Text key, IntWritable value) -> emit: (key, sum)
        private String pastConcatWords = null;                     // Record the Concat Words
        private int sumConcatWords = 0;
        private double score = 0;
        private int filterFrequencyLimit = 0;

        public void reduce(Text key, Iterable<IntWritable> values,
                           Context context
        ) throws IOException, InterruptedException {
            String concatFlagForPair = context.getConfiguration().get("concatFlagForPair");
            filterFrequencyLimit = Integer.parseInt(context.getConfiguration().get("filterFrequencyLimit"));

            String[] stringLists = key.toString().split(concatFlagForPair);
            String concatWords = stringLists[0];
            if (null == pastConcatWords) {
                pastConcatWords = concatWords;
            }
            // 统计相同concatWords的信息熵
            if (!concatWords.equals(pastConcatWords)) {
                if (sumConcatWords >= filterFrequencyLimit) {
                    score = -score / sumConcatWords + Math.log(sumConcatWords);
                    context.write(new Text(pastConcatWords), new Text(String.valueOf(score)));
                }
                // clear
                pastConcatWords = concatWords;
                sumConcatWords = 0;
                score = 0;
            }

            int count = 0;
            for (IntWritable val : values) {
                count += val.get();
            }
            sumConcatWords += count;
            score += Math.log(count) * count;
        }

        // 处理最后一组
        public void cleanup(Context context) throws IOException, InterruptedException {
            if (pastConcatWords != null && sumConcatWords >= filterFrequencyLimit) {
                score = score / sumConcatWords + Math.log(sumConcatWords);
                context.write(new Text(pastConcatWords), new Text(String.valueOf(score)));
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

    public static class PreWordSegmentEntropyOutputFormat extends TextOutputFormat {
        @Override
        public Path getDefaultWorkFile(TaskAttemptContext context, String extension) throws IOException {
            FileOutputCommitter committer = (FileOutputCommitter) getOutputCommitter(context);
            return new Path(committer.getWorkPath(), "PreWordSegmentEntropy.txt");
        }
    }

    public static class PostWordSegmentEntropyOutputFormat extends TextOutputFormat {
        @Override
        public Path getDefaultWorkFile(TaskAttemptContext context, String extension) throws IOException {
            FileOutputCommitter committer = (FileOutputCommitter) getOutputCommitter(context);
            return new Path(committer.getWorkPath(), "PostWordSegmentEntropy.txt");
        }
    }
}
