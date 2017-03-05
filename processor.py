import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import glob
import pickle

class Line:
    def __init__(self):
        self.detected = None
        pass

class FrameProcessor:
    def __init__(self, calib_dir = 'camera_cal/', nx = 9, ny = 6):
        self.left = Line()
        self.right = Line()
        self.calib_dir = calib_dir
        self.nx = nx
        self.ny = ny
        self.mtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None
        self.xsize = None
        self.ysize = None
        self.Minv = None
        self.ym_per_pix = 30/720 
        self.xm_per_pix = 3.7/700
        self.log = []
        #Processed frames counter
        self.counter = 0
        try:
            f = open('calibration.pickle', 'rb')
            self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = pickle.load(f)
        except IOError:
            self.calibrate()
                
    def calibrate(self):
        obp = np.zeros((self.nx*self.ny, 3), np.float32)
        obp[:, :2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1, 2)
        objpoints = []
        imgpoints = []
        for file in os.listdir(self.calib_dir):
            if file.endswith(".jpg"):
                img = cv2.imread(self.calib_dir + file)
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)
                if ret:
                    objpoints.append(obp)
                    imgpoints.append(corners)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        if ret:
            self.ret = ret
            self.mtx = mtx
            self.dist = dist
            self.rvecs = rvecs
            self.tvecs = tvecs
            with open('calibration.pickle', 'wb') as calib_file:
                pickle.dump([ret, mtx, dist, rvecs, tvecs], calib_file)
   
    def beye1(self, img):
        bottom_dx = self.xsize * 0.05
        top_dx = self.xsize * 0.4
        top_y = self.ysize * 2/3
        # source points - start from bottom left, clockiwse
        s1 = (bottom_dx, self.ysize)
        s2 = (top_dx, top_y)
        s3 = (self.xsize-top_dx, top_y)
        s4 = (self.xsize-bottom_dx, self.ysize)
        
        d1 = (0, self.ysize)
        d2 = (0, 0)
        d3 = (self.xsize, 0)
        d4 = (self.xsize, self.ysize)
        src = np.float32([s1, s2, s3, s4])
        dst = np.float32([d1, d2, d3, d4])
        M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src) 
        warped = cv2.warpPerspective(img, M, (self.xsize, self.ysize))
        return warped

    def mag_thresh(self, img, sobel_kernel=3, thresh=(0, 255)):
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag)/255 
        gradmag = (gradmag/scale_factor).astype(np.uint8) 
        binary_output = self.threshold(gradmag, thresh)
        return binary_output

    def abs_sobel_thresh(self, img, thresh, orient='x'):
        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        binary_output = self.threshold(scaled_sobel, thresh)
        return binary_output

    def dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi/2)):
        # Calculate the x and y gradients
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the gradient direction, 
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output =  self.threshold(absgraddir, thresh)

        # Return the binary image
        return binary_output

    def threshold(self, img, thresh):
        binary_output =  np.zeros_like(img)
        binary_output[(img >= thresh[0]) & (img <= thresh[1])] = 1
        return binary_output

    def pipe(self, img):
        undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        birds_eye = self.beye1(undist)
        hsv = cv2.cvtColor(birds_eye, cv2.COLOR_RGB2HSV)
        yellow_hsv_low  = np.array([ 0,  100,  100])
        yellow_hsv_high = np.array([ 80, 255, 255])
        yellow = cv2.inRange(hsv, yellow_hsv_low, yellow_hsv_high)
            
        white_hsv_low  = np.array([ 0,   0,   120])
        white_hsv_high = np.array([ 255,  80, 255])
        white = cv2.inRange(hsv, white_hsv_low, white_hsv_high)
            
        yellow_white = cv2.bitwise_or(white, yellow)
        gray = cv2.cvtColor(birds_eye, cv2.COLOR_BGR2GRAY)
        hls = cv2.cvtColor(birds_eye, cv2.COLOR_BGR2HLS)
        s_channel = hls[:,:,2]
        dsx = self.abs_sobel_thresh(gray, (20, 140), orient = 'x')
        mag = self.mag_thresh(s_channel, sobel_kernel = 5, thresh= (30, 100))
        s_threshold = self.threshold(s_channel, (180, 255))
        out = np.logical_or(dsx, s_threshold)
        return out, undist, birds_eye

    def analyze(self, img):
        undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        birds_eye = self.beye1(undist)
        gray = cv2.cvtColor(birds_eye, cv2.COLOR_BGR2GRAY)
        hls = cv2.cvtColor(birds_eye, cv2.COLOR_BGR2HLS)
        s_channel = hls[:,:,2]
        dsx = self.abs_sobel_thresh(s_channel, (20, 100), orient = 'x')
        s_threshold = self.threshold(s_channel, (180, 255))
        dsy = self.abs_sobel_thresh(s_channel, (20, 100), orient = 'y')
        mag = self.mag_thresh(s_channel, sobel_kernel = 5, thresh= (30, 100))
        ang = self.dir_threshold(s_channel, sobel_kernel = 15, thresh = (0.7, 1.3))
        return hls, s_threshold, dsx, dsy, mag, ang


    def slide(self, warped):
        x_middle = np.int(np.round(self.xsize/2))
        window_width = 50 
        n_layers = 9
        window_height = np.int(np.floor(self.ysize/n_layers))
        margin = 100 # How much to slide left and right for searching
        
        left_centroids = []
        right_centroids = []
        window = np.ones(window_width) # Create our window template that we will use for convolutions
        
        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template 
        
        # Sum quarter bottom of image to get slice, could use a different ratio
        left_quarter = warped[int(self.ysize * 3/4):, :x_middle]
        middle_y = int(self.ysize - window_height/2)
        l_sum = np.sum(left_quarter, axis=0)
        l_center = np.argmax(np.convolve(window, l_sum)) - window_width/2
        
        l_centroid = (l_center, middle_y)
        
        right_quarter = warped[int(self.ysize * 3/4):, x_middle:]
        r_sum = np.sum(right_quarter, axis=0)
        r_center = np.argmax(np.convolve(window,r_sum)) - window_width/2 + x_middle
        r_centroid = (r_center, middle_y)
        
        # Add what we found for the first layer
        left_centroids.append(l_centroid)
        right_centroids.append(r_centroid)
        
        # Threshold for convolution to avoid empty space centroids
        conv_threshold = 100
        # Go through each layer looking for max pixel locations
        
        for level in range(1, n_layers):
            # convolve the window into the vertical slice of the image
            #top, middle and bottom lines of the window
            bottom_y = int(self.ysize - level * window_height)
            top_y = bottom_y - window_height
            middle_y = bottom_y - window_height/2
            layer = warped[top_y:bottom_y, :]
            layer_sum = np.sum(layer, axis=0)
            conv_signal = np.convolve(window, layer_sum)
            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
            offset = window_width/2
            l_min_index = int(max(l_center 
                                + offset 
                                - margin, 0))
            l_max_index = int(min(l_center
                                + offset
                                + margin, self.xsize))
            if np.max(conv_signal[ l_min_index : l_max_index]) > conv_threshold:
                l_center = np.argmax(conv_signal[ l_min_index : l_max_index]) + l_min_index - offset
                l_centroid = (l_center, middle_y)
                left_centroids.append(l_centroid)
            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center
                                +offset
                                -margin, 0))
            r_max_index = int(min(r_center
                                +offset
                                +margin, self.xsize))
            if np.max(conv_signal[r_min_index : r_max_index]) > conv_threshold:
                r_center = np.argmax(conv_signal[r_min_index : r_max_index]) + r_min_index-offset
                r_centroid = (r_center, middle_y)
                right_centroids.append(r_centroid)
        
    
        left_centroids = np.array(left_centroids)
        right_centroids = np.array(right_centroids)
        left_fit = np.polyfit(left_centroids[:, 1], left_centroids[:, 0], 2)
        right_fit = np.polyfit(right_centroids[:, 1], right_centroids[:, 0], 2)
        return left_fit, right_fit, left_centroids, right_centroids
    
    def quadratic(self, fit, y):
        a = fit[0]
        b = fit[1]
        c = fit[2]
        out = a * np.square(y) + b * y +c
        return out

    def curve(self, fitx, ploty):
        fit = np.polyfit(ploty * self.ym_per_pix, fitx * self.xm_per_pix, 2)
        y = np.max(ploty) * self.ym_per_pix
        a = fit[0]
        b = fit[1]
        curve = np.power(1 + np.square(2*a*y +b), 1.5)/(2*np.abs(a))
        return curve

    def plot_all(self, warped, undist, left_fitx, right_fitx, ploty):
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 255))
        newwarp = cv2.warpPerspective(color_warp, self.Minv, (self.xsize, self.ysize))
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        return result
    
    def sanity(self, img, left_fit, right_fit, left_fitx, right_fitx, left_centroids, right_centroids):
        def calc_diff(line, xfit):
            if line.detected:
                diff = np.sum(np.abs(line.xfit - xfit))
            else:
                diff = None
            return diff

        left_diff = calc_diff(self.left, left_fitx)
        right_diff = calc_diff(self.right, right_fitx)
        self.left.detected = True
        self.right.detected = True
        self.left.xfit = left_fitx
        self.right.xfit = right_fitx

        cent_thresh = 4
        n_left_cent = len(left_centroids)
        n_right_cent = len(right_centroids)
        if (n_left_cent < cent_thresh | n_right_cent < cent_thresh):
            file_name = "fail_{:04d}.jpg".format(self.counter)
            bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite('failure_imgs/' + file_name, bgr)
        
        self.log.append((self.counter, left_diff, right_diff, 
        left_fit[0], left_fit[1], left_fit[2], 
        right_fit[0], right_fit[1], right_fit[2],
        n_left_cent, n_right_cent))
        
        return True
        
    def process(self, img):
        self.xsize = img.shape[1]
        self.ysize = img.shape[0]
        self.counter += 1
        warped, undist, birds_eye = self.pipe(img)
        left_fit, right_fit, left_centroids, right_centroids = self.slide(warped)
        
        ploty = np.linspace(0, self.ysize-1, self.ysize )
        
        left_fitx = self.quadratic(left_fit, ploty)
        right_fitx = self.quadratic(right_fit, ploty)

        left_curve = self.curve(left_fitx, ploty)
        right_curve = self.curve(right_fitx, ploty)
        _ = self.sanity(img, left_fit, right_fit, left_fitx, right_fitx, left_centroids, right_centroids)

        out = self.plot_all(warped, undist, left_fitx, right_fitx, ploty)
        curv_txt = "Curvature: {:6.0f}m`".format(left_curve)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(out, curv_txt, (10, 40), font, 1, (255, 255, 255), 2, cv2.LINE_AA )

        lane_center = left_fitx[-1] + (right_fitx[-1] - left_fitx[-1]) / 2
        car_center = np.int(self.xsize/2)
        deviation = (car_center - lane_center) * self.xm_per_pix
        dev_txt = "Deviation from lane center: {:4.2f}".format(deviation)
        cv2.putText(out, dev_txt, (int(self.xsize/2), 40), font, 1, (255, 255, 255), 2, cv2.LINE_AA )
        
        return out