import org.apache.commons.io.FileUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.File;
import java.io.IOException;

public class Main {

    public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
        Configuration conf = new Configuration();

        Job tempJob = Job.getInstance(conf, "preWordSegmentEntropyJob");
        tempJob.setJarByClass(PMI.class);
        tempJob.setMapperClass(PMI.PrePlusTokenizerMapper.class);
        tempJob.setCombinerClass(PMI.SumCombiner.class);
        tempJob.setPartitionerClass(PMI.ConcatWordsSamePartitioner.class);
        tempJob.setReducerClass(PMI.DivReducer.class);
        tempJob.setMapOutputKeyClass(Text.class);
        tempJob.setMapOutputValueClass(DoubleWritable.class);
        tempJob.setOutputKeyClass(Text.class);
        tempJob.setOutputValueClass(DoubleWritable.class);
        FileInputFormat.addInputPath(tempJob, new Path(args[0]));
        FileOutputFormat.setOutputPath(tempJob, new Path(GlobalSetting.tempPath));

        Job PMIJob = Job.getInstance(conf, "PMIJob");
        PMIJob.setJarByClass(PMI.class);
        PMIJob.setMapperClass(PMI.NormalMapper.class);
        PMIJob.setPartitionerClass(PMI.ConcatWordsSamePartitioner.class);
        PMIJob.setReducerClass(PMI.PMIReducer.class);
        PMIJob.setOutputKeyClass(Text.class);
        PMIJob.setOutputValueClass(DoubleWritable.class);
        FileInputFormat.addInputPath(PMIJob, new Path(GlobalSetting.tempPath + "/part-r-00000"));
        FileOutputFormat.setOutputPath(PMIJob, new Path(GlobalSetting.PMIPath));

        FileUtils.deleteDirectory(new File(GlobalSetting.tempPath));
        FileUtils.deleteDirectory(new File(GlobalSetting.PMIPath));

        if (tempJob.waitForCompletion(true)) {
            PMIJob.waitForCompletion(true);
        }

        // preWordEntropyJob
        Job preWordEntropyJob = Job.getInstance(conf, "preWordEntropyJob");
        preWordEntropyJob.setJarByClass(Entropy.class);
        preWordEntropyJob.setMapperClass(Entropy.PreTokenizerMapper.class);
        preWordEntropyJob.setCombinerClass(Entropy.SumCombiner.class);
        preWordEntropyJob.setPartitionerClass(Entropy.ConcatWordsSamePartitioner.class);
        preWordEntropyJob.setReducerClass(Entropy.FrequencyCombinationReducer.class);
        preWordEntropyJob.setMapOutputKeyClass(Text.class);
        preWordEntropyJob.setMapOutputValueClass(IntWritable.class);
        preWordEntropyJob.setOutputKeyClass(Text.class);
        preWordEntropyJob.setOutputValueClass(Text.class);
        preWordEntropyJob.setOutputFormatClass(Entropy.PreWordEntropyOutputFormat.class);
        FileInputFormat.addInputPath(preWordEntropyJob, new Path(args[0]));
        FileOutputFormat.setOutputPath(preWordEntropyJob, new Path(GlobalSetting.PreEntropyPath));

        // postWordEntropyJob
        Job postWordEntropyJob = Job.getInstance(conf, "postWordEntropyJob");
        postWordEntropyJob.setJarByClass(Entropy.class);
        postWordEntropyJob.setMapperClass(Entropy.PostTokenizerMapper.class);
        postWordEntropyJob.setCombinerClass(Entropy.SumCombiner.class);
        postWordEntropyJob.setPartitionerClass(Entropy.ConcatWordsSamePartitioner.class);
        postWordEntropyJob.setReducerClass(Entropy.FrequencyCombinationReducer.class);
        postWordEntropyJob.setMapOutputKeyClass(Text.class);
        postWordEntropyJob.setMapOutputValueClass(IntWritable.class);
        postWordEntropyJob.setOutputKeyClass(Text.class);
        postWordEntropyJob.setOutputValueClass(Text.class);
        postWordEntropyJob.setOutputFormatClass(Entropy.PostWordEntropyOutputFormat.class);
        FileInputFormat.addInputPath(postWordEntropyJob, new Path(args[0]));
        FileOutputFormat.setOutputPath(postWordEntropyJob, new Path(GlobalSetting.PostEntropyPath));

        FileUtils.deleteDirectory(new File(GlobalSetting.PreEntropyPath));
        preWordEntropyJob.submit();
        FileUtils.deleteDirectory(new File(GlobalSetting.PostEntropyPath));
        postWordEntropyJob.submit();

        boolean flag = preWordEntropyJob.isComplete() & postWordEntropyJob.isComplete();
        while (!flag) {
            flag = preWordEntropyJob.isComplete() & postWordEntropyJob.isComplete();
        }


        // preWordSegmentEntropyJob
        Job preWordSegmentEntropyJob = Job.getInstance(conf, "preWordSegmentEntropyJob");
        preWordSegmentEntropyJob.setJarByClass(Entropy.class);
        preWordSegmentEntropyJob.setMapperClass(Entropy.PreSegmentTokenizerMapper.class);
        preWordSegmentEntropyJob.setCombinerClass(Entropy.SumCombiner.class);
        preWordSegmentEntropyJob.setPartitionerClass(Entropy.ConcatWordsSamePartitioner.class);
        preWordSegmentEntropyJob.setReducerClass(Entropy.FrequencyCombinationReducer.class);
        preWordSegmentEntropyJob.setMapOutputKeyClass(Text.class);
        preWordSegmentEntropyJob.setMapOutputValueClass(IntWritable.class);
        preWordSegmentEntropyJob.setOutputKeyClass(Text.class);
        preWordSegmentEntropyJob.setOutputValueClass(Text.class);
        preWordSegmentEntropyJob.setOutputFormatClass(Entropy.PreWordSegmentEntropyOutputFormat.class);
        FileInputFormat.addInputPath(preWordSegmentEntropyJob, new Path(args[0]));
        FileOutputFormat.setOutputPath(preWordSegmentEntropyJob, new Path(GlobalSetting.PreSegmentEntropyPath));

        // postWordSegmentEntropyJob
        Job postWordSegmentEntropyJob = Job.getInstance(conf, "postWordSegmentEntropyJob");
        postWordSegmentEntropyJob.setJarByClass(Entropy.class);
        postWordSegmentEntropyJob.setMapperClass(Entropy.PostSegmentTokenizerMapper.class);
        postWordSegmentEntropyJob.setCombinerClass(Entropy.SumCombiner.class);
        postWordSegmentEntropyJob.setPartitionerClass(Entropy.ConcatWordsSamePartitioner.class);
        postWordSegmentEntropyJob.setReducerClass(Entropy.FrequencyCombinationReducer.class);
        postWordSegmentEntropyJob.setMapOutputKeyClass(Text.class);
        postWordSegmentEntropyJob.setMapOutputValueClass(IntWritable.class);
        postWordSegmentEntropyJob.setOutputKeyClass(Text.class);
        postWordSegmentEntropyJob.setOutputValueClass(Text.class);
        postWordSegmentEntropyJob.setOutputFormatClass(Entropy.PostWordSegmentEntropyOutputFormat.class);
        FileInputFormat.addInputPath(postWordSegmentEntropyJob, new Path(args[0]));
        FileOutputFormat.setOutputPath(postWordSegmentEntropyJob, new Path(GlobalSetting.PostSegmentEntropyPath));

        FileUtils.deleteDirectory(new File(GlobalSetting.PreSegmentEntropyPath));
        preWordSegmentEntropyJob.submit();
        FileUtils.deleteDirectory(new File(GlobalSetting.PostSegmentEntropyPath));
        postWordSegmentEntropyJob.submit();

        flag = preWordSegmentEntropyJob.isComplete() & postWordSegmentEntropyJob.isComplete();
        while (!flag) {
            flag = preWordSegmentEntropyJob.isComplete() & postWordSegmentEntropyJob.isComplete();
        }

        // Merge File to Score
        FileUtils.deleteDirectory(new File(GlobalSetting.MergePath));
        File MergeDirectory = new File(GlobalSetting.MergePath);
        assert MergeDirectory.mkdir();

        File PMIScoreFile = FileUtils.getFile(new File(GlobalSetting.PMIPath), "part-r-00000");
        File PreEntropyFile = FileUtils.getFile(
                new File(GlobalSetting.PreEntropyPath), GlobalSetting.inputFilesNameList[0]
        );
        File PostEntropyFile = FileUtils.getFile(
                new File(GlobalSetting.PostEntropyPath), GlobalSetting.inputFilesNameList[1]
        );

        FileUtils.copyFileToDirectory(PMIScoreFile, MergeDirectory);
        FileUtils.copyFileToDirectory(PreEntropyFile, MergeDirectory);
        FileUtils.copyFileToDirectory(PostEntropyFile, MergeDirectory);

        // Cache File: Segment Entropy
        FileUtils.deleteDirectory(new File(GlobalSetting.CachePath));
        File CacheDirectory = new File(GlobalSetting.CachePath);
        assert CacheDirectory.mkdir();

        File PreSegmentEntropyFile = FileUtils.getFile(
                new File(GlobalSetting.PreSegmentEntropyPath), GlobalSetting.inputFilesNameList[2]
        );
        File PostSegmentEntropyFile = FileUtils.getFile(
                new File(GlobalSetting.PostSegmentEntropyPath), GlobalSetting.inputFilesNameList[3]
        );
        FileUtils.copyFileToDirectory(PreSegmentEntropyFile, CacheDirectory);
        FileUtils.copyFileToDirectory(PostSegmentEntropyFile, CacheDirectory);


        Job scoreJob = Job.getInstance(conf, "ScoreCalculation");
        scoreJob.setJarByClass(Score.class);
        scoreJob.setMapperClass(Score.NormalMapper.class);
        scoreJob.setCombinerClass(Score.SumCombiner.class);
        scoreJob.setReducerClass(Score.SumReducer.class);
        scoreJob.setOutputKeyClass(Text.class);
        scoreJob.setOutputValueClass(DoubleWritable.class);
        FileInputFormat.addInputPath(scoreJob, new Path(GlobalSetting.MergePath));
        FileOutputFormat.setOutputPath(scoreJob, new Path(GlobalSetting.tempPath));

        Job sortJob = Job.getInstance(conf, "ScoreCalculation");
        sortJob.setJarByClass(Score.class);
        sortJob.setMapperClass(Score.SortMapper.class);
        sortJob.setReducerClass(Score.SortReducer.class);
        sortJob.setMapOutputKeyClass(DoubleWritable.class);
        sortJob.setMapOutputValueClass(Text.class);
        sortJob.setOutputKeyClass(Text.class);
        sortJob.setOutputValueClass(DoubleWritable.class);
        FileInputFormat.addInputPath(sortJob, new Path(GlobalSetting.tempPath));
        FileOutputFormat.setOutputPath(sortJob, new Path(args[1]));

        FileUtils.deleteDirectory(new File(GlobalSetting.tempPath));
        FileUtils.deleteDirectory(new File(args[1]));

        if (scoreJob.waitForCompletion(true)) {
            sortJob.waitForCompletion(true);
        }

        System.exit(0);
    }
}
