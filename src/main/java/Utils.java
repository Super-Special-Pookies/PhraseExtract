import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.wltea.analyzer.lucene.IKAnalyzer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;

class Utils {
    private final static String punctuationFlag = GlobalSetting.punctuationFlag;
    private static FileSystem fileSystem = null;

    static void setFileSystem(FileSystem fileSystem) {
        Utils.fileSystem = fileSystem;
    }

    static List<String> readLines(Path path) throws IOException {
        List<String> lineList = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new InputStreamReader(fileSystem.open(path)));
        String line;
        while ((line = reader.readLine()) != null) {
            lineList.add(line);
        }
        return lineList;
    }

    static boolean checkFileSystem() {
        return fileSystem == null;
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
