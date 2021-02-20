import org.apache.hadoop.io.Text;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.wltea.analyzer.lucene.IKAnalyzer;

import java.io.File;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

class Utils {
    private final static String punctuationFlag = GlobalSetting.punctuationFlag;

    private static boolean deleteFile(File dirFile) {
        if (!dirFile.exists()) {
            return false;
        }
        if (dirFile.isFile()) {
            return dirFile.delete();
        } else {
            for (File file : Objects.requireNonNull(dirFile.listFiles())) {
                deleteFile(file);
            }
        }
        return dirFile.delete();
    }


    private static List<String> getAnalyzedStr(Analyzer analyzer, String content) throws Exception {
        TokenStream stream = analyzer.tokenStream(null, new StringReader(content));
        CharTermAttribute term = stream.addAttribute(CharTermAttribute.class);
        stream.reset();
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

    static List<String> splitLine(Text value) {
        String line = replacePunctuationWithFlags(value.toString());
        Analyzer analyzer = new IKAnalyzer(true);
        List<String> splitText = null;
        try {
            splitText = getAnalyzedStr(analyzer, line);
        } catch (Exception e) {
            e.printStackTrace();
        }
        assert splitText != null;

        return splitText;
    }

    public static void main(String[] args) {
        List<String> res = splitLine(new Text("在新零售业态当中监守自盗\n"));
        for (String item : res) {
            System.out.println(item);
        }
    }
}
