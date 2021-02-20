final class GlobalSetting {
    final static String concatFlagForWord = "&";
    final static String concatFlagForPair = "#";
    final static String punctuationFlag = "符号";
    final static int filterFrequencyLimit = 0;
    final static String tempPath = "temp";
    final static String PMIPath = "PMI";
    final static String PreEntropyPath = "PreEntropy";
    final static String PostEntropyPath = "PostEntropy";
    final static String PreSegmentEntropyPath = "PreSegmentEntropy";
    final static String PostSegmentEntropyPath = "PostSegmentEntropy";
    final static String MergePath = "merge";
    final static String CachePath = "cache";

    final static String[] inputFilesNameList = new String[]{
            "PreWordEntropy.txt",
            "PostWordEntropy.txt",
            "PreWordSegmentEntropy.txt",
            "PostWordSegmentEntropy.txt"
    };

    final static String[] Paths = new String[]{
            PreEntropyPath,
            PostEntropyPath,
            PreSegmentEntropyPath,
            PostSegmentEntropyPath
    };

}
