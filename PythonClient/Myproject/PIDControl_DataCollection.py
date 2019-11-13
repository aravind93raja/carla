#!/usr/bin/env python3

from __future__ import print_function

import argparse
import logging
import random
import time
import matplotlib.pyplot as plt
from math import sqrt, atan2, sin, cos
import pandas as pd
import numpy as np
from scipy.interpolate import splprep, splev
import math
from carla.client import make_carla_client
from carla.sensor import Camera, Lidar
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line



def closest_waypoint(measurements_obj,waypoints_obj):
    location = np.array([
    measurements_obj.player_measurements.transform.location.x,
    measurements_obj.player_measurements.transform.location.y,
    ])
    dists = np.linalg.norm(waypoints_obj - location, axis=1)
    which_closest = np.argmin(dists)

    return which_closest,dists[which_closest]

def run_carla_client(args):                          
    frames_per_episode = 10000
    no_of_points=10000


    print(args.racetrack+'.txt')
    track_data = pd.read_csv(args.racetrack+'.txt', header = None)
    track_data=track_data/100
    waypoints_x_y = track_data.loc[:, [0, 1]].values
    tck, u = splprep(waypoints_x_y.T, u=None, s=2.0, per=1, k=3)
    u_new = np.linspace(u.min(), u.max(), no_of_points)
    x_new, y_new = splev(u_new, tck, der=0)
    waypoints_x_y=np.c_[x_new, y_new]

    with make_carla_client(args.host, args.port) as client:    
        print('CarlaClient connected')


        settings = CarlaSettings()
        settings.set(
        SynchronousMode=True,
        SendNonPlayerAgentsInfo=True,
        NumberOfVehicles=0,
        NumberOfPedestrians=0,
        WeatherId=1,
        QualityLevel=args.quality_level)

        camera1 = Camera('CameraDepth', PostProcessing='Depth')
        camera1.set_image_size(800, 600)
        camera1.set_position(0.30, 0, 1.30)
        settings.add_sensor(camera1)

        scene = client.load_settings(settings)
        
        number_of_player_start_pos=len(scene.player_start_spots)
        player_start = random.randint(0, max(0, number_of_player_start_pos - 1))
        print('Starting new episode...')
        client.start_episode(player_start)

        error_list=[]
        previous_cross_track_error=0
        depth_data_set1=np.empty((1,1,600,800))
        steering_dataset=np.empty((1,1))
        current_i=0
        frame=0

        previous_time=0

        while current_i<9990:

  
            measurements, sensor_data = client.read_data()
            print(sensor_data['CameraDepth'].data)
            if frame>=500:
                depth_array = sensor_data['CameraDepth'].data
                filename='./data/valid/datax/'+str(frame-499)
                np.save(filename,depth_array)

            

            for name, measurement in sensor_data.items():
                filename = args.out_filename_format.format(1,name, frame)
                print(filename)
                measurement.save_to_disk(filename)
            
            current_time=measurements.game_timestamp
            dt = float(current_time - previous_time)/1000
            current_v=measurements.player_measurements.forward_speed * 3.6

            current_i,cross_track_error=closest_waypoint(measurements,waypoints_x_y)
            print(current_i)
            
            current_x=measurements.player_measurements.transform.location.x
            current_y=measurements.player_measurements.transform.location.y
            


            
            determine_side=(current_x-waypoints_x_y[current_i,0])*(waypoints_x_y[current_i+1,1]-waypoints_x_y[current_i,1])-(current_y-waypoints_x_y[current_i,1])*(waypoints_x_y[current_i+1,0]-waypoints_x_y[current_i,0])
            direction=math.copysign(1,determine_side)

            cross_track_error=direction*cross_track_error
            cross_track_error_change=cross_track_error-previous_cross_track_error
            error_list.append(cross_track_error)

            error_derivative=cross_track_error_change/dt
        

            
            
            steer_output=(cross_track_error*0.6+0.1*error_derivative)

            if frame>=500:
                steering_array=np.reshape(steer_output,(1,1))
                filename='./data/valid/datay/'+str(frame-499)
                np.save(filename,steering_array)
                
            frame=frame+1

 

            client.send_control(
                        steer=steer_output,
                        throttle=0.6,
                        brake=0.0,
                        hand_brake=False,
                        reverse=False)
            previous_time=current_time
            previous_cross_track_error=cross_track_error


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-l', '--lidar',
        action='store_true',
        help='enable Lidar')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Epic',
        help='graphics quality level, a lower level makes the simulation run considerably faster.')
    argparser.add_argument(
        '-i', '--images-to-disk',
        action='store_true',
        dest='save_images_to_disk',
        help='save images (and Lidar data if active) to disk')
    argparser.add_argument(
        '-c', '--carla-settings',
        metavar='PATH',
        dest='settings_filepath',
        default=None,
        help='Path to a "CarlaSettings.ini" file')
    argparser.add_argument(
        '-r', '--racetrack',
        dest='racetrack',
        default=None,
        help='Racetrack Number')

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    args.out_filename_format = '_out/episode_{:0>4d}/{:s}/{:0>6d}'

    while True:
        try:

            run_carla_client(args)

            print('Done.')
            return

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
