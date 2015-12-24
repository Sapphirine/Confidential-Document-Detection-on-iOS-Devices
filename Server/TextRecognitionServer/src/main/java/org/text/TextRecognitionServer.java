package org.text;
 
import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.apache.commons.io.IOUtils;

import com.amazonaws.services.s3.AmazonS3;
import com.amazonaws.services.s3.AmazonS3Client;
import com.amazonaws.services.s3.model.GetObjectRequest;
import com.amazonaws.services.s3.model.S3Object;
import com.amazonaws.services.s3.model.S3ObjectInputStream;

public class TextRecognitionServer extends HttpServlet
{
    /**
	 * Text Recognition Server on AWS EC2
	 */
	private static final long serialVersionUID = 9087580376653616292L;
	AmazonS3 s3Client;
	
	public void init() throws ServletException {
		s3Client = new AmazonS3Client();
	}
	
	protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException
    {
        response.setContentType("text/html");
        response.setStatus(HttpServletResponse.SC_OK);
        response.getWriter().println("<h1>Text Recognition Server is working...</h1>");
        response.getWriter().println("session = " + request.getSession(true).getId());
    }
	
	protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException
    {
        String bucketName = request.getParameter("bucketName");
        String keyName = request.getParameter("keyName");
        String userID = request.getParameter("userID");
        GetObjectRequest getObjectRequest = new GetObjectRequest(bucketName, keyName);
        S3Object object = s3Client.getObject(getObjectRequest);
        S3ObjectInputStream objectContent = object.getObjectContent();
        String imagePath = "/image/" + userID + "_" + keyName;
        IOUtils.copy(objectContent, new FileOutputStream(imagePath));
        String text = executeCommand("/src/textRecognition " + imagePath);
        String[] strs = text.split("###Result:");
        if (strs.length < 2) {
        	response.setStatus(HttpServletResponse.SC_BAD_REQUEST);
        	return;
        }
        response.setStatus(HttpServletResponse.SC_OK);
        String textPath = imagePath + ".txt";
        PrintWriter out = new PrintWriter(textPath);
        out.print(strs[1]);
        out.close();
        String result = executeCommand("/src/classify " + textPath);
        response.getWriter().println(result);
    }
	
	private String executeCommand(String command) {

		StringBuffer output = new StringBuffer();

		Process p;
		try {
			p = Runtime.getRuntime().exec(command);
			p.waitFor();
			BufferedReader reader = 
					new BufferedReader(new InputStreamReader(p.getInputStream()));
			String line = "";			
			while ((line = reader.readLine())!= null) {
				output.append(line + "\n");
			}

		} catch (Exception e) {
			e.printStackTrace();
		}

		return output.toString();

	}

}
