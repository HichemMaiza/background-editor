import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.opencv.core.CvType.CV_32FC1;

public class BGTranslator implements Translator<Mat, Mat> {

    public static final long[] UNET_INPUT_SHAPE = {3, 320, 320};
    private static final Logger LOGGER = LoggerFactory.getLogger(BGTranslator.class);
    private static final int U2NET_ROWS = 320 ;
    private static final int U2NET_COLS = 320 ;

    /**
     * Processes the output NDList to the corresponding output object.
     *
     * @param ctx  the toolkit used for post-processing
     * @param list the output NDList after inference, usually immutable in engines like
     *             PyTorch. @see <a href="https://github.com/deepjavalibrary/djl/issues/1774">Issue 1774</a>
     * @return the output object of expected type
     * @throws Exception if an error occurs during processing output
     */

    @Override
    public Mat processOutput(TranslatorContext ctx, NDList list) throws Exception {

        NDArray ndArray = list.get(0);
        float[] yData = ndArray.toFloatArray();

        Mat outputImage = new Mat(U2NET_ROWS, U2NET_COLS, CV_32FC1);
        outputImage.put(0, 0, yData);

        // Convert the Mat to 8-bit format (if necessary)
        Mat outputImage8bit = new Mat();

        // max value
        Core.MinMaxLocResult minMaxLocResult = Core.minMaxLoc(outputImage) ;

        double ma = minMaxLocResult.maxVal ;
        double mi = minMaxLocResult.minVal ;

        Core.subtract(outputImage, new Scalar(mi), outputImage);
        Core.divide(outputImage, new Scalar(ma-mi), outputImage);

        outputImage.convertTo(outputImage8bit, CvType.CV_8U, 255.0);

        return outputImage8bit;
    }

    /**
     * Processes the input and converts it to NDList.
     *
     * @param ctx   the toolkit for creating the input NDArray
     * @param input the input object
     * @return the {@link NDList} after pre-processing
     * @throws Exception if an error occurs during processing input
     */
    @Override
    public NDList processInput(TranslatorContext ctx, Mat input) throws Exception {

        Mat resizedInput = new Mat();
        if ((input.cols() != U2NET_COLS) || (input.cols() != U2NET_ROWS)){
            LOGGER.info("resizing the input image to {}X{}", U2NET_ROWS, U2NET_COLS);
            Imgproc.resize(input, resizedInput, new Size(U2NET_ROWS, U2NET_COLS));
        }else {
            resizedInput = input ;
        }

        byte[] bytes = new byte[(int) (resizedInput.total() * resizedInput.channels())];
        resizedInput.get(0, 0, bytes);

        // Convert bytes to floats and normalize
        float[] floats = convertBytesToFloats(bytes);

        NDManager ndManager = ctx.getNDManager();
        NDArray ndArray = ndManager.create(floats) ;

        NDArray reshaped =  ndArray.reshape(UNET_INPUT_SHAPE) ;
        return new NDList(reshaped);
    }

    public static float[] convertBytesToFloats(byte[] bytes) {

        // Convert bytes to floats and normalize

        float[] floats = new float[bytes.length];

        float[] mean = {0.485f, 0.456f, 0.406f};
        float[] std = {0.229f, 0.224f, 0.225f};

        int numPixels = floats.length / 3;
        float maxValue = 255.0f ;
        for (int i = 0; i < numPixels; i++) {

            // Convert each set of three bytes to a float
            int index = i * 3;

            // Extract RGB values from consecutive bytes
            int red = bytes[index] & 0xFF;
            int green = bytes[index + 1] & 0xFF ;
            int blue = bytes[index + 2] & 0xFF ;

            // Convert RGB values to float and add to the list
            float redFloat = (float) red / maxValue ;
            float greenFloat = (float) green / maxValue;
            float blueFloat = (float) blue / maxValue;

            float redNormalized = (redFloat - mean[0]) / std[0] ;
            float greenNormalized = (greenFloat - mean[1]) / std[1] ;
            float blueNormalized = (blueFloat - mean[2]) / std[2] ;

            floats[i] = redNormalized ;
            floats[i + numPixels] = greenNormalized;
            floats[i + 2 * (numPixels)] = blueNormalized;
        }
        return floats;
    }

}
