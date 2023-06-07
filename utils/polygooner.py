import numpy as np
import cv2

# ============================================================================
FINAL_LINE_COLOR = (31, 95, 255)
WORKING_LINE_COLOR = (127, 127, 127)

# ============================================================================

def PolyArea(points):
    x = points[:,0]
    y = points[:,1]
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def draw_grid(img, pxstep, pystep, line_color=(0, 255, 0), thickness=1, type_=cv2.LINE_AA):
    '''(ndarray, 3-tuple, int, int) -> void
    draw gridlines on img
    line_color:
        BGR representation of colour
    thickness:
        line thickness
    type:
        8, 4 or cv2.LINE_AA
    pxstep:
        grid line frequency in pixels
    '''
    x = pxstep
    y = pystep
    while x < img.shape[1]:
        cv2.line(img, (x-2, 0), (x-2, img.shape[0]), color=line_color, lineType=type_, thickness=thickness)
        x += pxstep

    while y < img.shape[0]:
        cv2.line(img, (0, y), (img.shape[1], y), color=line_color, lineType=type_, thickness=thickness)
        y += pystep

class PolygonDrawer(object):
    def __init__(self, window_name, image):
        self.window_name = window_name # Name for our window

        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] #[(50,50), (50,100), (100,100), (100,50)]#[] # List of points defining our polygon
        self.image = image
        self.im_dim = (image.shape[0], image.shape[1])
        self.area = 0.0
        self.intermed = []

    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)

        if self.done: # Nothing more to do
            return

        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = [x, y]
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points

            xf = int(np.floor((x//(256/13)) * (256/13)))
            yf = int(np.floor((y//(144/6)) * (144/6)))
            xc = int(np.ceil(xf + 256/13))
            yc = int(np.ceil(yf + 144/6))

            self.intermed.append([xf, yf])
            self.intermed.append([xc, yf])
            self.intermed.append([xc, yc])
            self.intermed.append([xf, yc])

            if self.points:
                xM = max(np.array(self.intermed)[:,0])
                xm = min(np.array(self.intermed)[:,0])
                yM = max(np.array(self.intermed)[:, 1])
                ym = min(np.array(self.intermed)[:, 1])

                self.points = [[xm, ym],[xM, ym],[xM, yM],[xm, yM]]

            else:
                self.points = self.intermed

            print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))            # self.done = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click means we're done
            print("Completing polygon with %d points." % len(self.points))
            self.done = True


    def run(self):
        # Let's create our working window and set a mouse callback to handle events
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        draw_grid(self.image, 20, 24)
        cv2.imshow(self.window_name, self.image)
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        self.points = []

        while(not self.done):
            # This is our drawing loop, we just continuously draw new images
            # and show them in the named window
            canvas = self.image
            if (len(self.points) > 0):
                # Draw all the current polygon segments
                cv2.polylines(canvas, np.array([self.points]), False, FINAL_LINE_COLOR, 1)
                # And  also show what the current segment would look like
                # cv2.line(canvas, self.points[-1], self.current, WORKING_LINE_COLOR)
            # Update the window
            cv2.imshow(self.window_name, canvas)
            cv2.resizeWindow(self.window_name, self.im_dim[1]*9, self.im_dim[0]*9)
            # And wait 50ms before next iteration (this will pump window messages meanwhile)
            if cv2.waitKey(50) == 27: # ESC hit
                self.done = True

        # User finished entering the polygon points, so let's make the final drawing
        #canvas = np.zeros(self.im_dim, np.uint8)
        # of a filled polygon
        if (len(self.points) > 0):
            cv2.polylines(canvas, np.array([self.points]),True, FINAL_LINE_COLOR)
            self.area = PolyArea(np.array(self.points))
        # And show it
        cv2.imshow(self.window_name, canvas)
        # Waiting for the user to press any key
        cv2.waitKey()

        cv2.destroyWindow(self.window_name)
        return canvas


# ============================================================================

# if __name__ == "__main__":
#     img = "/home/makramchahine/Desktop/Guardian_long_demo/1664396689.82_ctrnn_wiredcfccell_fine_train/1664396689.796.png"
#     img = cv2.imread(img)
#     pd = PolygonDrawer("Polygon",img)
#     image = pd.run()
#     cv2.imwrite("polygon.png", image)
#     print("Polygon = %s" % pd.points)
#     print("area = %d" % pd.area)
