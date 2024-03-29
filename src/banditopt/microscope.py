
'''
This module implements wrapper functions to access and modify easily the STED
parameters through specpy.
'''

import time
import numpy
import pickle

import pyautogui
def click_on_screenshot(screenshot_png):
    app_logo = (numpy.array(pyautogui.locateOnScreen(screenshot_png, grayscale=True))).astype(int)
    pyautogui.click(x=app_logo[0], y=app_logo[1], clicks=1, interval=0, button='left')


try:
    from specpy import Imspector

    im = Imspector()
    measurement = im.active_measurement()
except ModuleNotFoundError as err:
    print(err)
    print("Calling these functions might raise an error.")


def get_config(message=None, image=None):
    '''Fetch and return the active configuration in Imspector.

    :param message: If defined, print the following message.

    :returns: The active configuration (specpy Configuration object).
    '''
    if message is not None:
        print(message)
    print("Manually select imaging configuration then press enter.")
    print(image)
    if image is not None:
        click_on_screenshot("auto_gui/Imspector_logo.png")
        click_on_screenshot("auto_gui/"+image)
        click_on_screenshot("auto_gui/promp_logo.png")
        # click_on_screenshot("auto_gui/setting_conf_logo.png")
        pyautogui.typewrite(['enter', 'enter'])
    input()
    return measurement.active_configuration()


def clone(conf):
    '''Clones the corresponding configuration in the measurement, activates and returns the clone.

    :param conf: A configuration object.
    :param name: Name of the clone measurement

    :returns: A configuration object
    '''
    return measurement.clone(conf)


def get_params(conf):
    '''Fetch and return the parameters of a configuration object.

    :param conf: A configuration object.

    :returns: A dict of parameters.
    '''
    return conf.parameters("ExpControl")

def get_linestep(conf, step_id):
    line_steps = conf.parameters("ExpControl/measurement/line_steps")
    return line_steps['repetitions']


def get_power(conf, laser_id, channel_id=0):
    '''Fetch and return the power of a laser in a specific configuration.

    :param conf: A configuration object.
    :param laser_id: ID of the laser in Imspector (starting from 0).

    :returns: The power (%).
    '''
    # params = conf.parameters("ExpControl/measurement")
    #TODO: should we return a ratio instead?
    return conf.parameters(f"ExpControl/measurement/channels/{channel_id}/lasers/{laser_id}/power/calibrated")


def get_pixelsize(conf):
    '''Fetch and return the pixel size in a specific configuration.

    :param conf: A configuration object.

    :returns: Tuple of (x, y) pixel sizes (m).
    '''
    x = conf.parameters("ExpControl/scan/range/x/psz")
    y = conf.parameters("ExpControl/scan/range/y/psz")
    return x, y


def get_resolution(conf):
    '''Fetch and return the resolution in a specific configuration.

    :param conf: A configuration object.

    :returns: Tuple of (x, y) resolutions (m).
    '''
    x = conf.parameters("ExpControl/scan/range/x/res")
    y = conf.parameters("ExpControl/scan/range/y/res")
    return x, y


def get_imagesize(conf):
    '''Fetch and return the image size in a specific configuration.

    :param conf: A configuration object.

    :returns: Tuple of (x, y) image sizes (m).
    '''
    x = conf.parameters("ExpControl/scan/range/x/len")
    y = conf.parameters("ExpControl/scan/range/y/len")
    return x, y


def get_offsets(conf):
    '''Fetch and return the offsets in a specific configuration.

    :param conf: A configuration object.

    :returns: Tuple of (x, y) offsets.
    '''
    x = conf.parameters("ExpControl/scan/range/x/off")
    y = conf.parameters("ExpControl/scan/range/y/off")
    return x, y


def get_dwelltime(conf,channel_id=0):
    ''' Fetch and return the pixel dwell time in a specific configuration.

    :param conf: A configuration object.

    :returns: The dwell time (s).
    '''
    params = conf.parameters("ExpControl/measurement/pixel_steps")
    return params['step_duration'][channel_id]


def get_overview(conf, prefix="Overview ", name=None):
    if name is None:
        print(prefix)
        print("Type the name of the overview then press enter.")
        overview = prefix + str(input())
    else:
        overview = prefix + name
    print('overview')
    #import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()
    return conf.stack(overview).data()[0][0]


def get_image(conf):
    return conf.stack("Confocal_635").data()[0][0]


def set_pixelsize(conf, x, y=None):
    '''Sets the pixel size

    :param conf: Configuration window
    :param x: pixel size in x
    :param y: pixel size in y'''
    conf.set_parameters("ExpControl/scan/range/x/psz", x)
    if y is None:
        conf.set_parameters("ExpControl/scan/range/y/psz", x)
    else:
        conf.set_parameters("ExpControl/scan/range/y/psz", y)


def set_offsets(conf, x, y):
    '''Set the offsets in a specific configuration.

    :param conf: A configuration object.
    :param x: The x offset.
    :param y: The y offset.
    '''
    conf.set_parameters("ExpControl/scan/range/x/off", x)
    conf.set_parameters("ExpControl/scan/range/y/off", y)


def set_imagesize(conf, width, height):
    '''Set the imagesize in a specific configuration.

    :param conf: A configuration object.
    :param width: width (m)
    :param height: height (m)
    '''
    # print("The function receives {}, which is of type {}... It should be of type {}".format(resolution, type(resolution[0]), type(get_resolution(conf)[0])))
    conf.set_parameters("ExpControl/scan/range/x/len", width)
    conf.set_parameters("ExpControl/scan/range/y/len", height)
    # print("The resolution of the window is now : {}. It should be {} {}".format(get_imagesize(conf), width, height))


def set_numberframe(conf, num):
    '''Set the number of frame in a xyt configuration.

    :param conf: A configuration object.
    :param num: The number of frame.
    '''
    conf.set_parameters("ExpControl/scan/range/t/res", num)


# def set_power(conf, power, laser_id,channel_id=0):
#     '''Set the power of a laser in a specific configuration.
#
#     :param conf: A configuration object.
#     :param laser_id: ID of the laser in Imspector (starting from 0).
#     :param power: Power of the laser in [0, 1].
#     '''
#     params = conf.parameters("ExpControl/measurement")
#
#     if laser_id == 0:
#         print('405 nm')
#         params["channels"][channel_id]["lasers"][laser_id]["power"]["calibrated"] = power*1e-3
#     else:
#         print('all good')
#         params["channels"][channel_id]["lasers"][laser_id]["power"]["calibrated"] = power
#     conf.set_parameters("ExpControl/measurement", params)
# def set_power(conf, power, laser_id):
#     '''Set the power of a laser in a specific configuration.
#
#     :param conf: A configuration object.
#     :param int laser_id: Imdex of the laser in Imspector (starting from 0).
#     :param float power: Power of the laser in [0, 1].
#     '''
#     lasers = conf.parameters('ExpControl/lasers/power_calibrated')
#     lasers[laser_id]["value"]["calibrated"] = power * 100
#     conf.set_parameters("ExpControl/lasers/power_calibrated", lasers)
import time
def set_power(conf, power, laser_id, channel_id=0):
    '''Set the power of a laser in a specific configuration.

    :param conf: A configuration object.
    :param int laser_id: Imdex of the laser in Imspector (starting from 0).
    :param float power: Power of the laser in [0, 1].
    '''
    conf.set_parameters(f"ExpControl/measurement/channels/{channel_id}/lasers/{laser_id}/power/calibrated", power)
# import pyperclip
# def set_power(conf, power, laser_id):
#     print('WARNING: THIS JUST SETS THE STED POW WITH PYAUTOGUI')
#     click_on_screenshot("auto_gui/Imspector_logo.png")
#     click_on_screenshot("auto_gui/sted_logo.png")
#     pow_loc = (numpy.array(pyautogui.locateOnScreen("auto_gui/pow_logo.png", grayscale=True))).astype(int)
#     if laser_id==5:
#         pyautogui.moveTo(pow_loc[0]+100, pow_loc[1]-10)
#     elif laser_id==6:
#         pyautogui.moveTo(pow_loc[0]+100, pow_loc[1]+10)
#     elif laser_id==3:
#         pyautogui.moveTo(pow_loc[0]+100, pow_loc[1]-57)
#     pyperclip.copy(str(power))
#     pyautogui.click()
#     pyautogui.click()
#     pyautogui.hotkey('ctrl', 'v')
#     click_on_screenshot("auto_gui/promp_logo.png")







def set_dwelltime(conf, dwelltime, channel_id=0):
    '''Set the pixel dwell time in a specific configuration.

    :param conf: A configuration object.
    :param dwelltime: Pixel dwell time (s).
    '''
    params = conf.parameters("ExpControl/measurement/pixel_steps")
    params['step_duration'][channel_id]=dwelltime
    conf.set_parameters("ExpControl/measurement/pixel_steps", params)



def activate_linestep(conf, status=True):
    params = conf.parameters("ExpControl/measurement")
    params["line_steps"]["active"]=status
    conf.set_parameters("ExpControl/measurement", params)
def activate_pixelstep(conf, status=True):
    params = conf.parameters("ExpControl/measurement")
    params["pixel_steps"]["active"]=status
    conf.set_parameters("ExpControl/measurement", params)

def set_linestep(conf, linestep, step_id):
    '''Set the line step of a specific channel in a specific configuration.

    :param conf: A configuration object.
    :param linestep: Line step.
    :param step_id: ID of the line step in Imspector (starting from 0).
    '''
    if numpy.round(linestep) != linestep:
        linestep = numpy.round(linestep)
        print("WARNING!!!!!!!!!!!!!!!!!! THE LINESTEP WAS JUST ROUNDED FROM A FLOAT VALUE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    conf.set_parameters("ExpControl/measurement/line_steps/repetitions", linestep)





def set_frametrig(conf,state):
    ''' Set the state of the frame trigger,

    :param conf: A configuration object
    :param state: Boolean
    '''
    trigger = conf.parameters("ExpControl/trigger")
    trigger["frametrig_use"] = state
    conf.set_parameters("ExpControl/trigger", trigger)


def set_chans_on(conf, line_id, step_id):
    '''
    '''
    chans_on = conf.parameters("ExpControl/gating/linesteps/chans_on")
    chans_on[line_id][step_id] = True
    conf.set_parameters("ExpControl/gating/linesteps/chans_on", chans_on)

####  RESCue Parameters   ###

def set_rescue_signal_level(conf, signal_level, channel_id):
    '''Set the RESCue signal level in a specific configuration.

    :param conf: A configuration object.
    :param signal_level: Signal level of RESCue.
    :param channel_id: ID of the RESCue channel in Imspector (starting from 0).
    '''
    channels = conf.parameters("ExpControl/rescue/channels")
    channels[channel_id]["signal_level"] = signal_level
    conf.set_parameters("ExpControl/rescue/channels", channels)


def set_rescue_strength(conf, strength, channel_id):
    '''Set the RESCue strength in a specific configuration.

    :param conf: A configuration object.
    :param strength: Strength of RESCue.
    :param channel_id: ID of the RESCue channel in Imspector (starting from 0).
    '''
    channels = conf.parameters("ExpControl/rescue/channels")
    channels[channel_id]["strength"] = strength
    conf.set_parameters("ExpControl/rescue/channels", channels)



def set_uth_manual(conf, channel_id):
    channels = conf.parameters("ExpControl/rescue/channels")
    channels[channel_id]["UTh_manual"] = True
    channels[channel_id]["UTh_use"] = True

    conf.set_parameters("ExpControl/rescue/channels", channels)

def set_uth_thresh(conf, threshold, channel_id):
    channels = conf.parameters("ExpControl/rescue/channels")
    channels[channel_id]["UTh_threshold"] = threshold
    conf.set_parameters("ExpControl/rescue/channels", channels)

def set_uth_auto(conf, channel_id):
    channels = conf.parameters("ExpControl/rescue/channels")
    channels[channel_id]["UTh_manual"] = False

    conf.set_parameters("ExpControl/rescue/channels", channels)

def turn_uth_off1(conf, channel_id):
    channels = conf.parameters("ExpControl/rescue/channels")
    channels[channel_id]["UTh_manual"] = True
    #channels[channel_id]["UTh_use"] = False
    conf.set_parameters("ExpControl/rescue/channels", channels)
def turn_uth_off2(conf, channel_id):
    channels = conf.parameters("ExpControl/rescue/channels")
    #channels[channel_id]["UTh_manual"] = True
    channels[channel_id]["UTh_use"] = False
    conf.set_parameters("ExpControl/rescue/channels", channels)

def set_LTh_auto(conf,channel_id):
    channels = conf.parameters("ExpControl/rescue/channels")
    channels[channel_id]["set_thresholds_manually"] = False
    conf.set_parameters("ExpControl/rescue/channels", channels)

def set_LTh_manual(conf, channel_id):
    channels = conf.parameters("ExpControl/rescue/channels")
    channels[channel_id]["set_thresholds_manually"] = True
    conf.set_parameters("ExpControl/rescue/channels", channels)

def set_LTh_numtimes(conf, numtime, channel_id):
    channels = conf.parameters("ExpControl/rescue/channels")

    channels[channel_id]["num_LTh"] = numtime
    conf.set_parameters("ExpControl/rescue/channels", channels)

def set_LTh_thresh(conf, thresholds, channel_id):
    channels = conf.parameters("ExpControl/rescue/channels")
    channels[channel_id]["LTh"] = thresholds
    conf.set_parameters("ExpControl/rescue/channels", channels)


def set_LTh_times(conf, times, channel_id):
    channels = conf.parameters("ExpControl/rescue/channels")
    channels[channel_id]["LTh_times"] = times
    conf.set_parameters("ExpControl/rescue/channels", channels)

def turn_on_rescue(conf, channel_id):
    channels = conf.parameters("ExpControl/rescue/channels")
    channels[channel_id]["on"] = True
    conf.set_parameters("ExpControl/rescue/channels", channels)


def turn_off_rescue(conf, channel_id):
    channels = conf.parameters("ExpControl/rescue/channels")
    channels[channel_id]["on"] = False
    channels[channel_id]["rescue_allowed"] = False
    conf.set_parameters("ExpControl/rescue/channels", channels)

def set_auto_blanking(conf, channel_id):
    channels = conf.parameters("ExpControl/rescue/channels")
    channels[channel_id]["auto_blank"] = True
    conf.set_parameters("ExpControl/rescue/channels", channels)

def set_manual_blanking(conf,channel_id):
    channels = conf.parameters("ExpControl/rescue/channels")
    channels[channel_id]["auto_blank"] = False
    conf.set_parameters("ExpControl/rescue/channels", channels)

def set_manual_blanking_lasers(conf,lasers,channel_id):
    blanklasers=[False,False,False,False,False,False,False,False]
    for laser in lasers:
        blanklasers[laser]=True
    channels = conf.parameters("ExpControl/rescue/channels")
    channels[channel_id]["blanking"] = blanklasers
    conf.set_parameters("ExpControl/rescue/channels", channels)

def turn_probe_on(conf, channel_id):
    channels = conf.parameters("ExpControl/rescue/channels")
    channels[channel_id]['use_as_probe'] = True
    conf.set_parameters("ExpControl/rescue/channels", channels)

def turn_probe_off(conf, channel_id):
    channels = conf.parameters("ExpControl/rescue/channels")
    channels[channel_id]['use_as_probe'] = False
    conf.set_parameters("ExpControl/rescue/channels", channels)
###   End of RESCue parameters   ####


def acquire(conf):
    '''Activate the given configuration and acquire an image stack.

    :param conf: A configuration object.

    :return: An image stack (3d array) and the acquisition time (seconds).
    '''
    measurement.activate(conf)
    start = time.time()
    im.run(measurement)
    end = time.time()
    stacks = [conf.stack(i) for i in range(conf.number_of_stacks())]
    x, y = get_offsets(conf)
    print("Acquiring with configuration", conf.name(), "at offset x:", x, ", y:", y)

    #conf.stack(conf.name())
    # chop the first 2 lines because of imaging problems I guess
    # chop 0.08 seconds because life
    return [[image[2:].copy() for image in stack.data()[0]] for stack in stacks], end - start - 0.08

def acquire_saveasmsr(conf,savepath):
    '''Activate the given configuration and acquire an image stack.

    :param conf: A configuration object.
    :param savepath: a path to a file

    :return: An image stack (3d array) and the acquisition time (seconds).
    '''
    measurement.activate(conf)
    start = time.time()
    im.run(measurement)
    end = time.time()
    stacks = [conf.stack(i) for i in range(conf.number_of_stacks())]
    x, y = get_offsets(conf)
    print("Acquiring with configuration", conf.name(), "at offset x:", x, ", y:", y)
    measurement.save_as(savepath,True)
    #conf.stack(conf.name())
    # chop the first 2 lines because of imaging problems I guess
    # chop 0.08 seconds because life
    return [[image[2:].copy() for image in stack.data()[0]] for stack in stacks],end - start - 0.08

if __name__ == "__main__":
    import pickle

    params = get_params()
    # with open("test", "wb") as f:
        # pickle.dump(params, f)
    # with open("test", "rb") as f:
        # params = pickle.load(f)
    # set_params(params)
