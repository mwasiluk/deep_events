from __future__ import print_function

import json
import os
import random as rn
import sys
# import pickle
from cStringIO import StringIO
from optparse import OptionParser
import tempfile
from liner2 import Liner2, start_jvm, create_relation

from deep_events import run as run_deep_events

class Pipe(object):
    def __init__(self, config, debug = True):
        self.debug = debug
        self.config = config

    def run(self, input_file, output_file, pipe_name=None, input_format="txt", output_format=None):
        last_output=None
        batch = False
        single_input_format = input_format
        if input_format.startswith("batch:"):
            batch = True
            single_input_format = input_format[len("batch:"):]
            if not output_format:
                output_format = "batch:cclrel"
        if not output_format:
            output_format = "cclrel"
        print("output_format", output_format)

        if not pipe_name:
            pipe_name = self.config["default_pipe"]

    
        if input_format == "txt":
            print("Running concraft")
            (_, morph_file) = tempfile.mkstemp()
            if self.config['concraft']['use_server']:
                concraft_cmd = self.config['concraft']['bin'] + ' client --port ' + self.config['concraft']['server_port'] + ' -n < ' + input_file + ' > ' + morph_file
            else:
                concraft_cmd = self.config['concraft']['bin'] + ' tag ' + self.config['concraft']['model'] + ' < ' + input_file + ' > ' + morph_file
                # print(concraft_cmd)
    
            os.system(concraft_cmd)

            (_, last_output) = tempfile.mkstemp(suffix='.xml')
            os.system(self.config['corpus_get'] + ' -i plain -o ccl -t nkjp ' + morph_file + ' > ' + last_output)
    
            input_format = "ccl"
    
        elif single_input_format == "ccl" or single_input_format == "cclrel":
            last_output = input_file
        else:
            print("input format not supported!")
            exit()
    
        start_jvm([self.config['maltparser']['jar'], self.config['liner']['jar']], [self.config['liner']['lib'], self.config['maltparser']['lib']])

        pipe_conf = self.config["pipes"][pipe_name]
        next_out_format = "cclrel"
        if batch:
            next_out_format = "batch:"+next_out_format

        for i, c in enumerate(pipe_conf):

            if i < len(pipe_conf)-1:
                (last_output, input_format) = self.run_sub_config(c, input_file=last_output, input_format=input_format, output_format=next_out_format)
            else:
                (last_output, input_format) = self.run_sub_config(c, input_file=last_output, input_format=input_format, out_file=output_file, output_format=output_format)


    def run_sub_config(self, config_name, input_file, input_format="cclrel", output_format="cclrel", out_file=None):
        liner_prefix = 'liner.'
        if config_name.startswith(liner_prefix):
            return self.run_liner(config_name[len(liner_prefix):], input_file=input_file, input_format=input_format, output_format=output_format, out_file=out_file)

        return self.run_deep_events_config(config_name, input_file=input_file, input_format=input_format, output_format=output_format, out_file=out_file)
    
    def run_deep_events_config(self, config_name, input_file, input_format="cclrel", output_format="cclrel", out_file=None):
        if not out_file:
            out_dir = tempfile.mkdtemp()
            out_file = os.path.join(out_dir, 'out.xml')
    
        if self.debug:
            print(config_name, out_file)
    
        run_deep_events(load_config(self.config[config_name]["config"]), input_file, out_file,
                        input_format=input_format, output_format=output_format, model_file=self.config[config_name]["model"])
        return out_file, output_format
    
    
    def run_liner(self, config_name, input_format, output_format, input_file, out_file = None):
        if not out_file:
            out_dir = tempfile.mkdtemp()
            out_file = os.path.join(out_dir, 'out.xml')

        if self.debug:
            print("liner", config_name, out_file)
    
        os.system("java -Djava.library.path=" + self.config['liner']['lib'] + " -jar " + self.config['liner']['jar'] + ' pipe' +
                  ' -i ' + input_format +
                  ' -o ' + output_format +
                  ' -m ' + self.config['liner']['configs'][config_name] +
                  ' -f ' + input_file +
                  ' -t ' + out_file)
        return out_file, output_format

def load_config(filename):
    with open(filename) as json_data:
        c = json.load(json_data)
    return c


def go():
    parser = OptionParser(usage="Tool for event detection")
    parser.add_option('-c', '--config', type='string', action='store', default='config/events_pipe_config.json',
                      dest='config',
                      help='json config file location')
    parser.add_option('-i', '--input-format', type='string', action='store',
                      dest='input_format',
                      help='input format', default="txt")
    parser.add_option('-o', '--output-format', type='string', action='store',
                      dest='output_format',
                      help='output format, default same as input format', default=None)
    parser.add_option('-p', '--pipe', type='string', action='store',
                      dest='pipe_name',
                      help='name of pipe configuration to use', default=None)
    (options, args) = parser.parse_args()
    fn_output = None


    if len(args) != 2:
        print('Need to provide input and output.')
        print('See --help for details.')
        sys.exit(1)
    fn_input, fn_output = args

    config = load_config(options.config)

    # if not options.output_format:
    #     options.output_format = options.input_format

    Pipe(config).run(fn_input, fn_output, input_format=options.input_format, output_format = options.output_format, pipe_name=options.pipe_name)

if __name__ == '__main__':
    go()