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

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class Score {
    private final static String tempPath = GlobalSetting.tempPath;

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        String mergeFilesPath = args[0];

        FileSystem fileSystem = FileSystem.get(new URI(args[0]), conf);
        Utils.setFileSystem(fileSystem);

        Job scoreJob = Job.getInstance(conf, "ScoreCalculation");
        scoreJob.setJarByClass(Score.class);
        scoreJob.setMapperClass(NormalMapper.class);
        scoreJob.setCombinerClass(SumCombiner.class);
        scoreJob.setReducerClass(SumReducer.class);
        scoreJob.setOutputKeyClass(Text.class);
        scoreJob.setOutputValueClass(DoubleWritable.class);
        scoreJob.addCacheFile(new URI(
                new Path(GlobalSetting.PreSegmentEntropyPath, GlobalSetting.inputFilesNameList[2]).toString()));
        scoreJob.addCacheFile(new URI(
                new Path(GlobalSetting.PostSegmentEntropyPath, GlobalSetting.inputFilesNameList[3]).toString()));
        FileInputFormat.addInputPath(scoreJob, new Path(mergeFilesPath));
        FileOutputFormat.setOutputPath(scoreJob, new Path(tempPath));

        Job sortJob = Job.getInstance(conf, "ScoreCalculation");
        sortJob.setJarByClass(Score.class);
        sortJob.setMapperClass(SortMapper.class);
        sortJob.setReducerClass(SortReducer.class);
        sortJob.setMapOutputKeyClass(DoubleWritable.class);
        sortJob.setMapOutputValueClass(Text.class);
        sortJob.setOutputKeyClass(Text.class);
        sortJob.setOutputValueClass(DoubleWritable.class);
        FileInputFormat.addInputPath(sortJob, new Path(tempPath));
        FileOutputFormat.setOutputPath(sortJob, new Path(args[1]));

        fileSystem.delete(new Path(GlobalSetting.tempPath), true);
        fileSystem.delete(new Path(args[1]), true);
        if (scoreJob.waitForCompletion(true)) {
            sortJob.waitForCompletion(true);
        }
        System.exit(0);
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
        // Combine: (Text key, DoubleWritable value) -> emit: (key, sum)
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

    public static class SumReducer
            extends Reducer<Text, DoubleWritable, Text, DoubleWritable> {
        HashMap<String, Double> PreSegmentEntropyMap = new HashMap<>();
        HashMap<String, Double> PostSegmentEntropyMap = new HashMap<>();
        // Combine: (Text key, DoubleWritable value) -> emit: (key, sum)
        private DoubleWritable result = new DoubleWritable();

        private List<String> readCache(Configuration conf, URI localCacheFile) throws IOException {
            List<String> res = new ArrayList<>();
            BufferedReader reader = new BufferedReader(new InputStreamReader(
                    FileSystem.get(conf).open(new Path(localCacheFile.getPath()))));
            String line;
            while ((line = reader.readLine()) != null) {
                res.add(line);
            }
            return res;
        }

        public void setup(Context context)
                throws IOException, InterruptedException, NullPointerException {
            super.setup(context);
            Configuration conf = context.getConfiguration();
            URI PreSegmentEntropyFile = context.getCacheFiles()[0];
            URI PostSegmentEntropyFile = context.getCacheFiles()[1];
            List<String> PreSegmentEntropy = readCache(conf, PreSegmentEntropyFile);
            List<String> PostSegmentEntropy = readCache(conf, PostSegmentEntropyFile);

            for (String line : PreSegmentEntropy) {
                String[] tempSplitList = line.split("\t");
                PreSegmentEntropyMap.put(tempSplitList[0], Double.parseDouble(tempSplitList[1]));
            }

            for (String line : PostSegmentEntropy) {
                String[] tempSplitList = line.split("\t");
                PostSegmentEntropyMap.put(tempSplitList[0], Double.parseDouble(tempSplitList[1]));
            }
        }

        public void reduce(Text key, Iterable<DoubleWritable> values,
                           Context context
        ) throws IOException, InterruptedException {
            String[] tempSplitList = key.toString().split(GlobalSetting.concatFlagForWord);
            double sum = 0;
            for (DoubleWritable val : values) {
                sum += val.get();
            }
            if (PreSegmentEntropyMap.containsKey(tempSplitList[0])) {
                sum -= PreSegmentEntropyMap.get(tempSplitList[0]);
            }
            if (PostSegmentEntropyMap.containsKey(tempSplitList[1])) {
                sum -= PostSegmentEntropyMap.get(tempSplitList[1]);
            }

            result.set(sum);
            context.write(key, result);
        }
    }


    public static class SortMapper
            extends Mapper<Object, Text, DoubleWritable, Text> {
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] list = value.toString().split("\t");
            String word = list[0];
            double val = Double.parseDouble(list[1]);
            context.write(new DoubleWritable(-val), new Text(word));
        }
    }

    public static class SortReducer
            extends Reducer<DoubleWritable, Text, Text, DoubleWritable> {
        public void reduce(DoubleWritable key, Iterable<Text> values,
                           Context context
        ) throws IOException, InterruptedException {
            for (Text val : values) {
                context.write(val, new DoubleWritable(key.get() * -1));
            }
        }
    }
}
