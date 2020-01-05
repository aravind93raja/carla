#!/usr/bin/env python2
from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    

import tensorflow as tf
import argparse
import logging
import random
import time

from carla.client import make_carla_client
from carla.sensor import Camera, Lidar
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError

from keras.models import load_model
import numpy as np




def run_carla_client(args):

    model = load_model('/home/deeplearner/UnrealEngine_4.18/carla/PythonClient/Dataset/MAE_O.02545.hdf5')
    model.summary()

    with make_carla_client(args.host, args.port) as client:
        print('CarlaClient connected')
    
        settings = CarlaSettings()
        settings.set(
            SynchronousMode=True,
            SendNonPlayerAgentsInfo=False,
            NumberOfVehicles=0,
            NumberOfPedestrians=0,
            WeatherId=1,
            QualityLevel=args.quality_level
        )

        settings.randomize_seeds()

        camera = Camera('CameraDepth', PostProcessing='Depth', FOV=69.4)      
        camera.set_image_size(50,50)
        camera.set_position(0.45, 0, 1.55)
        settings.add_sensor(camera)

        scene = client.load_settings(settings)

        num_of_player_starts = len(scene.player_start_spots)
        player_start = random.randint(0, max(0, num_of_player_starts - 1))

        print('Starting new episode...', )
        client.start_episode(player_start)

        frame=0
        time_end=0

        while frame<=10099:
            time_start=time.time()
            if time_start-time_end>=.05:

                measurements, sensor_data = client.read_data()

                depth_array = np.log(sensor_data['CameraDepth'].data).astype('float32')
                print(frame)
                depth_array=np.reshape(depth_array,(1,50,50,1))
                s=model.predict(depth_array)
                s=s[0]
 
                frame=frame+1
                
                client.send_control(
                            steer=s,
                            throttle=0.6,
                            brake=0.0,
                            hand_brake=False,
                            reverse=False)
                time_end=time_start


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
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Epic',
        help='graphics quality level, a lower level makes the simulation run considerably faster.')
    argparser.add_argument(
        '-c', '--carla-settings',
        metavar='PATH',
        dest='settings_filepath',
        default=None,
        help='Path to a "CarlaSettings.ini" file')

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
