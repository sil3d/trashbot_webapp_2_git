## Setup Instructions

### Prerequisites

Ensure you have the following software installed to get started:

- [DB Browser for SQLite](https://sqlitebrowser.org/) for database management
- [Autodesk Fusion 360](https://www.autodesk.com/products/fusion-360/overview) for 3D design
- [Arduino IDE](https://www.arduino.cc/en/software) for motor control
- [PuTTY](https://www.putty.org/) for SSH connection with Raspberry Pi
- [Raspberry Pi Imager](https://www.raspberrypi.com/software/) to flash SD cards
- [Git](https://git-scm.com/) and [GitHub](https://github.com/) to download the code
-[vscode](https://code.visualstudio.com/) to write you code

### Environment Setup

1. **Create a new Conda environment with Python 3.9:**
   ```bash
   conda create --name myenv python=3.9
   conda activate myenv
   ```

2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

all required knowledge




*********** HOW IT WORK ***************

**** images are save into static/captures_images *** and them we juste save this location into the databse , simple !
best alternative solution save the image itsefl into the database , now we gonna use the LONGBLOG the save (long text very long text , because image it's just 1 and 0 et here the image will be convert into base64 file for html ect to see into u well page....)


1 - detect image
2- classifie and save into database
3-rotate motor

- [click_with_crtl_left_mouse_to_open](static/thumbnails/Screenshot_222412.png)
- [index](static/thumbnails/Screenshot_222430.png)
- [history](static/thumbnails/Screenshot_222446.png)
- [history](static/thumbnails/Screenshot_222458.png)