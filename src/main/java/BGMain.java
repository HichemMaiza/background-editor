import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import nu.pattern.OpenCV;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;


import java.io.IOException;
import java.nio.file.Paths;

public class BGMain {

    public static final String MODEL_PATH =  "src/test/resources/u2net_enriched.onnx";
    public static final String TEST_IMAGE_PATH = "src/test/resources/elonmusk.jpg";

    public static final String OUTPUT_IMAGE_PATH = "src/test/resources/elonmusk_output.jpg";
    public static final String OUTPUT_MASK_PATH = "src/test/resources/elonmusk_output_mask.jpg";


    static {
        OpenCV.loadLocally();
    }

    public static void main(String[] args) throws ModelNotFoundException, MalformedModelException, IOException {

        Mat inputImage = Imgcodecs.imread(TEST_IMAGE_PATH) ;

        Criteria<Mat, Mat> criteria = Criteria.builder()
                .optEngine("OnnxRuntime")
                .setTypes(Mat.class, Mat.class)
                .optTranslator(new BGTranslator())
                .optModelPath(Paths.get(MODEL_PATH))
                .optDevice(Device.gpu())
                .optProgress(new ProgressBar())
                .build();

        try (ZooModel<Mat, Mat> model = criteria.loadModel();
             Predictor<Mat, Mat> rembg = model.newPredictor()) {
             Mat mask = rembg.predict(inputImage);
             Mat postProcess = postProcess(inputImage, mask);

             Imgcodecs.imwrite(OUTPUT_IMAGE_PATH, postProcess);
             Imgcodecs.imwrite(OUTPUT_MASK_PATH, mask);

             HighGui.namedWindow("Mask", HighGui.WINDOW_NORMAL);
             HighGui.imshow("Mask", mask);

             HighGui.namedWindow("Input Image", HighGui.WINDOW_NORMAL);
             HighGui.imshow("Input Image", inputImage);

             HighGui.namedWindow("Output Result", HighGui.WINDOW_NORMAL);
             HighGui.imshow("Output Result", postProcess);
             HighGui.waitKey();

        } catch (TranslateException e) {
            throw new RuntimeException(e);
        }
    }

    public static Mat postProcess(Mat inputImage, Mat mask){

        Mat blurMask = new Mat() ;
        Mat binaryMask = new Mat() ;
        Mat resultImage = new Mat() ;
        Imgproc.GaussianBlur(mask, blurMask, new Size(3,3), 0, 0);

        Imgproc.resize(blurMask, blurMask, new Size(inputImage.cols(), inputImage.rows()));
        Imgproc.threshold(blurMask, binaryMask, 140, 255, Imgproc.THRESH_BINARY);

        inputImage.copyTo(resultImage, binaryMask);

        return resultImage;
    }

}