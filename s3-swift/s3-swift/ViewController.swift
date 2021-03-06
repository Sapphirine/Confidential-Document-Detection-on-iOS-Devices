//
//  ViewController.swift
//  s3-swift
//
//  Created by Barrett Breshears on 12/6/14.
//  Copyright (c) 2014 Barrett Breshears. All rights reserved.
//

import UIKit
import AssetsLibrary


class ViewController: UIViewController, UINavigationControllerDelegate, UIImagePickerControllerDelegate {

    @IBOutlet var selectedImage:UIImageView?
    var imagePickerController:UIImagePickerController?
    var loadingBg:UIView?
    var progressView:UIView?
    var progressLabel:UILabel?
    
    var uploadRequest:AWSS3TransferManagerUploadRequest?
    var filesize:Int64 = 0
    var amountUploaded:Int64 = 0
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    //MARK: S3 stuff
    func uploadToS3(){
        
        // get the image
        let img:UIImage = selectedImage!.image!
        
        // create a local image that we can use to upload to s3
        let path:NSString = (NSTemporaryDirectory() as NSString).stringByAppendingPathComponent("image.png")
        let imageData:NSData = UIImagePNGRepresentation(img)!
        imageData.writeToFile(path as String, atomically: true)
        
        // once the image is saved we can use the path to create a local fileurl
        let url:NSURL = NSURL(fileURLWithPath: path as String)
        
        // next we set up the S3 upload request manager
        uploadRequest = AWSS3TransferManagerUploadRequest()
        // set the bucket
        uploadRequest?.bucket = "bigdata-v1"
        // I want this image to be public to anyone to view it so I'm setting it to Public Read
        uploadRequest?.ACL = AWSS3ObjectCannedACL.PublicRead
        // set the image's name that will be used on the s3 server. I am also creating a folder to place the image in
        uploadRequest?.key = "test/image.png"
        // set the content type
        uploadRequest?.contentType = "image/png"
        // and finally set the body to the local file path
        uploadRequest?.body = url;
        
        // we will track progress through an AWSNetworkingUploadProgressBlock
        uploadRequest?.uploadProgress = {[unowned self](bytesSent:Int64, totalBytesSent:Int64, totalBytesExpectedToSend:Int64) in
            
            dispatch_sync(dispatch_get_main_queue(), { () -> Void in
                self.amountUploaded = totalBytesSent
                self.filesize = totalBytesExpectedToSend;
                self.update()

            })
        }
        
        // now the upload request is set up we can creat the transfermanger, the credentials are already set up in the app delegate
        let transferManager:AWSS3TransferManager = AWSS3TransferManager.defaultS3TransferManager()
        transferManager.upload(uploadRequest).continueWithBlock {[unowned self]
            task -> AnyObject in
            
            if(task.error != nil){
                NSLog("%@", task.error!);
            }else{ // if there aren't any then the image is uploaded!
                // this is the url of the image we just uploaded
                NSLog("https://s3.amazonaws.com/bigdata-v1/test/image.png");
            }
            
            self.removeLoadingView()
            return "all done";
        }
    }
    
    func update(){
        let percentageUploaded:Float = Float(amountUploaded) / Float(filesize) * 100
        progressLabel?.text = NSString(format:"Uploading: %.0f%%", percentageUploaded) as String
    }
    
    // Mark: camera and IBAction stuff
    
    @IBAction func cameraBtnClicked(sender: UIButton){
        imagePickerController = UIImagePickerController()
        imagePickerController?.delegate = self
        imagePickerController?.sourceType = UIImagePickerControllerSourceType.Camera
        self.presentViewController(imagePickerController!, animated: true, completion: nil)
    }
    
    @IBAction func galleryBtnClicked(sender: UIButton){
//        var alertController:UIAlertController?
//        alertController = UIAlertController(title: "Analysis Result: ",
//            message: "This image contains confidential information, and will NOT be saved!",
//            preferredStyle: .Alert)
//        
//        self.presentViewController(alertController!,
//            animated: true,
//            completion: nil)
        let alertController = UIAlertController(title: "Analysis Result: ", message: "This image contains confidential information, and will NOT be saved!", preferredStyle: UIAlertControllerStyle.Alert)
        alertController.addAction(UIAlertAction(title)
        self.presentedViewController(alertController, animated: true, completion: nil)
        imagePickerController = UIImagePickerController()
        imagePickerController?.delegate = self
        imagePickerController?.sourceType = UIImagePickerControllerSourceType.PhotoLibrary
        self.presentViewController(imagePickerController!, animated: true, completion: nil)
    }
    
//    @IBAction func uploadBtnClicked(sender: UIButton){
//        self.createLoadingView()
//        self.uploadToS3()
//    }
    
    func imagePickerController(picker: UIImagePickerController,
      didFinishPickingMediaWithInfo info: [String : AnyObject]){
        picker.dismissViewControllerAnimated(true, completion: nil)
        selectedImage?.image = (info["UIImagePickerControllerOriginalImage"] as! UIImage)
        self.createLoadingView()
        self.uploadToS3()
    }
    
    func createLoadingView(){
        loadingBg = UIView(frame: self.view.frame)
        loadingBg?.backgroundColor = UIColor(red: 0, green: 0, blue: 0, alpha: 0.35)
        self.view.addSubview(loadingBg!)
        
        progressView = UIView(frame: CGRectMake(0, 0, 250, 50))
        progressView?.center = self.view.center
        progressView?.backgroundColor = UIColor.whiteColor()
        loadingBg?.addSubview(progressView!)
        
        progressLabel = UILabel(frame: CGRectMake(0, 0, 250, 50))
        progressLabel?.textAlignment = NSTextAlignment.Center
        progressView?.addSubview(progressLabel!)
        progressLabel?.text = "Uploading:";
    }
    
    func removeLoadingView(){
        loadingBg?.removeFromSuperview()
    }

}

