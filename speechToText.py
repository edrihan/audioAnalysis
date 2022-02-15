#!/usr/bin/env python3
#from vosk import Model, KaldiRecognizer
from vosk import Model, KaldiRecognizer, SetLogLevel
import json
"example.wav"
import sys
import shutil
from tqdm import tqdm
import os
import wave
import subprocess
import queue
import threading
from tkinter.messagebox import showinfo
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
from tkinter import *
import ffmpeg
import pydub
import sounddevice as sd
import pyautogui # For pressing keyboard keys
import configparser
#from pynput.keyboard import Key, Listener
import logging
import pynput

import global_hotkeys # pip install pypiwin32

class Utility:
    @classmethod
    def getClassVariables(cls,Class):
        return {key: value for key, value in Class.__dict__.items() if not key.startswith('__') and not callable(key)}

    @classmethod
    def saveToFile(cls, text: str, filePath):
        print(cls.makeDirectory(''.join(filePath.replace('/','\\').split('\\')[:-1])))
        with open(filePath, "w+", encoding="utf-8") as file:
            file.write(text)
            file.close()
            print("wrote {} chars to ////file:{}".format(len(text),filePath))

    @classmethod
    def makeDirectory(cls, path, printSuccess=True, printSkip=False):
        if os.path.isdir(path):
            if printSkip:
                print("skipped making {} because it already exists.".format(path))
            return True
        os.makedirs(path, 0o777)
        if printSuccess:
            print("Successfully created the directory %s " % path)
    @classmethod
    def delete_file(cls,filepath):
        try:
            os.remove(filepath)
        except FileNotFoundError:
            print('skipping deleting',filepath,'as it does not exist.')
class SonicAnnotator:
    examplePlugins = (#'vamp:vamp-example-plugins:fixedtempo:tempo',
                        #'vamp:vamp-example-plugins:fixedtempo:candidates',
                      #'vamp:vamp-example-plugins:percussiononsets:detectionfunction',
                      #'vamp:silvet:silvet:notes',
                        #'vamp:vamp-aubio:aubionotes:notes',

                        #'vamp:qm-vamp-plugins:qm-onsetdetector',

                        #'vamp:qm-vamp-plugins:qm-transcription',
                        'vamp:nnls-chroma:chordino:chordnotes',
                        'vamp:nnls-chroma:chordino:simplechord',
                        'vamp:nnls-chroma:nnls-chroma:basschroma',
                        #'vamp:nnls-chroma:chordino:harmonicchange',
                        #'vamp:vamp-example-plugins:percussiononsets:detectionfunction'

                        #'vamp:silvet:silvet:pitchactivation',
                        'vamp:vamp-aubio:aubionotes:notes',

                        #'vamp:tipic:tipic:pitch',
                      'vamp:qm-vamp-plugins:qm-keydetector:tonic',
                        'vamp:vamp-aubio:aubiotempo:beats',
                        'vamp:qm-vamp-plugins:qm-barbeattracker:bars'
                      )
    libPath = "sonic-annotator"
    filepath = 'a3.wav'
    filepath = '01 Say What You Will.wav'
    filepath = '01 - The Art of the Fugue, BWV 1080- Contrapunctus I.flac'
    filepath = r"01 - Chunga's Revenge.flac"
    #filepath = '07 Mercy, Mercy, Mercy.flac'
    env = os.environ
    @classmethod
    def threaded(cls,function):
        # Call work function
        t1 = threading.Thread(target=function)
        t1.start()

    @classmethod
    def saveTransforms(cls):
        '''sonic-annotator -s vamp:vamp-example-plugins:fixedtempo:tempo > test.n3'''
        '''sonic-annotator -t test.n3 audio.wav -w csv --csv-stdout'''
        print(cls.getAvailableTransforms())
        input('holy baloney')


    @classmethod
    def getAvailableTransforms(cls):
        process = subprocess.Popen([cls.libPath, '-l'],
                                   stdout=subprocess.PIPE, env=cls.env)
        return process.stdout.read().decode('utf-8').split(os.linesep)


    @classmethod
    def applyTransform(cls,filename:str,saveToCSV=False,skip_if_file_exists=False):
        def function():
            windowsFix = '.cmd'


            #os.system('set VAMP_PATH=/path/to/plugin/directory')



            #for transform in tqdm(cls.getAvailableTransforms()):
            results = []
            for transform in tqdm(cls.examplePlugins):
                print('making transform',transform)
                #transform = 'vamp:vamp-example-plugins:fixedtempo:tempo'
                #doIt = input("Do it? ({})".format(transform))

                output_filename = '.'.join(filename.split('.')[:-1]) + '.mid'
                Utility.delete_file(output_filename)
                print('$'*100+output_filename)
                new_filename = '.'.join(filename.split('.')[:-1]) +'_' +transform.replace(':','_')+'.mid'
                if os.path.isfile(new_filename) and skip_if_file_exists:
                    continue#break
                process = subprocess.Popen(
                    ['sonic-annotator', '-d', transform, filename.replace('.mid',''), '-w', 'midi', '--midi-force',
                      ], env=cls.env, stdout=subprocess.PIPE)
                process.wait()
                #Utility.saveToFile(process.stdout.read().decode('utf-8'),new_filename.replace('.flac',''))
                #process.stdout = open(new_filename.replace('.flac',''),'w+')
                #print(process.stdout.read())
                #process.stdout.close()
                #process.wait()
                print(('copied \n{} \n\tto \n{}'.format(output_filename, new_filename)))
                shutil.copyfile(output_filename,new_filename)
                #os.rename(output_filename,new_filename)

                #results.append(process.stdout.read().decode('utf-8'))

                #To be able to make the file, NOW that it exists
                #process.wait()
                not_done = True



                print('there',filename,output_filename)


                if saveToCSV:
                    #output_filename = ''.join(filename.split('.')[0:-1]) #+ '.csv'
                    output_filename = ''.join(filename.split('.')[0:-1]) #+ '.csv'
                    #Utility.delete_file(output_filename)
                    print('here',filename,output_filename)
                    '''process = subprocess.Popen(
                        ['sonic-annotator', '-d', transform, filename, '-w', 'csv', '--csv-stdout','--csv-force', '"'+output_filename+'"'],
                        env=cls.env, stdout=subprocess.PIPE)'''
                    process = subprocess.Popen(['sonic-annotator', '-d', transform, filename, '-w', 'csv', '--csv-force'],
                                               env=cls.env)
                    #process = subprocess.Popen(['sonic-annotator', '-d', transform, cls.filepath, '-w', 'csv',],env=cls.env)

                    '''process = subprocess.Popen(['libpath', '-loglevel', 'quiet', '-i',
                                                    filepath.replace('/', '\\'),
                                                    '-ar', str(sample_rate), '-ac', '1', '-f', 's16le', '-'],
                                                   stdout=subprocess.PIPE)'''


            print('we are done\n\n\n{}'.format('\n'.join([cls.examplePlugins[r] +': '+ results[r] for r in range(len(results))])))
        function()
        #subprocess.call()
print('\n'.join(SonicAnnotator.getAvailableTransforms()))
#SonicAnnotator.applyTransform()



print('loading model')
model = Model("model")
print('finished loading model')
modelHD = False
def threaded(function):
    # Call work function
    t1 = threading.Thread(target=function)
    t1.start()

def makeHDModel():
    global modelHD, model
    print('loading HD model')
    modelHD = Model("vosk-model-en-us-0.22")
    print('finished loading HD model. replacing model')
    #model=modelHD
    #open_button.config(text="Open Files (HD)")
def compileApp():
    import PyInstaller.__main__

    PyInstaller.__main__.run([
        'my_script.py',
        '--onefile',
        '--windowed'
    ])


threaded(makeHDModel)
class Options:
    speech_to_text_enabled = False
    update_textbox_enabled = True
    save_to_text_file = False
    speech_to_text_alive = False
    speech_to_typing_enabled = True
    sample_rate = 16000
    saved_settings = (update_textbox_enabled,)
    config = configparser.ConfigParser()
    config['DEFAULT'] = {'speech_to_text_enabled': 'False',
                         'update_textbox_enabled': 'True',
                         'save_to_text_file': False,
                        'speech_to_text_alive':False,
                        'speech_to_typing_enabled':True,
                        'sample_rate' :16000,
                         }
    config['Sonic_Annotator'] = {
        'libPath': "sonic-annotator"
    }
    with open('config.ini', 'w') as configfile:
        config.write(configfile)

    @classmethod
    def saveSettings(cls):
        with open('config.json', 'w+') as f:
            print(Utility.getClassVariables(Options),'hey there')
            json.dump(Utility.getClassVariables(Options), f)

    @classmethod
    def loadSettings(cls):
        print('loading settings')
        config = configparser.ConfigParser()
        f = open("config.ini",'r')
        settings = config.read(f)


        file1 = open('config.ini', 'r')
        Lines = file1.readlines()
        for line in Lines:
            print(line)
            if '[' in line:
                section = line.replace('[','').replace(']','')
            elif '=' in line:
                setting = line.split(' = ')[0]
                value = line.split(' = ')[1]
                if section.strip() == 'Sonic_Annotator':
                    print('loaded setting',setting,section,value)
                    setattr(SonicAnnotator,setting,value)
                else:
                    print('section is',section,'not Sonic_Annotator')

        '''for section in config:
            print(section,settings,config.sections())
            for key in config[section]:
                print(key)
                if key in cls.saved_settings:
                    if section == 'Sonic_Annotator':
                        print('loaded setting',key,section[key])
                        SonicAnnotator[key] = section'''

    #['example.ini']
    #>> > config.sections() config.read('example.ini')


        '''
            # Opening JSON file
        f = open('config.json')

        # returns JSON object as
        # a dictionary
        config = json.load(f)

        print(config['emp_details'])
        for setting in config.keys():
            if setting in cls.saved_settings:
                cls.setting = config[setting]'''

Options.loadSettings()
class WordSubs:
    invalidResults = ('','huh',', ','the')
    commandTrigger = 'make'
    commands = {
        'bracket': '(',
        'period': '. ',
        'dot': '.',
        'rest': ', ',
        'line break': '\n',
        'space': ' ',
        'tab': '    ',
        'dope thing': '----'
    }
    multiWordCommandNames = [i for i in commands.keys() if ' ' in i]
    subs = {' i ': ' I '}

    @classmethod
    def applyWordCommands(cls,result: str) -> str:
        trig = WordSubs.commandTrigger
        commandNames = WordSubs.commands.keys()
        commands = WordSubs.commands
        if trig in result:
            words = result.split(' ')

            #print('words == ',words,'result == ',result)
            try:
                commandIdx = words.index(trig) + 1
            except:
                raise TypeError(str(words) + ' ' +str(result))
            didMakeSub = False
            multiWordName = False
            for w,word in enumerate(words[commandIdx:]):
                try:
                    nextWord = words[commandIdx + w + 1]
                except:
                    nextWord = False#print('end of road',words,commandIdx,w)
                #input(words)
                #if any([word in cWord.split(' ') for cWord in cls.multiWordCommandNames]):


                for m,multiWord in enumerate(cls.multiWordCommandNames):
                    #print('\n',m,multiWord,word,nextWord)
                    if nextWord and word + ' ' + nextWord in multiWord:
                        words[w + commandIdx] = word + ' ' + nextWord


                        multiWordName = word + ' ' + nextWord


                if multiWordName and word == multiWordName.split(' ')[0] and nextWord == multiWordName.split(' ')[1]:
                    words[w+commandIdx] = commands[multiWordName]
                    didMakeSub = True


                elif words[w + commandIdx] in commandNames:

                    words[w+commandIdx] = commands[word]


                    didMakeSub = True
                else:
                    break
            if didMakeSub:
                #print('before', words)
                words.__delitem__(commandIdx - 1)
                #print('after', words)
                #print(multiWordName)
                #input('stuff')
                if multiWordName:

                    #print('before',words,multiWordName)
                    words.__delitem__(commandIdx)
                    multiWordNameWordCount = multiWordName.count(' ') + 1

                    '''print("1: "+' '.join(words[:commandIdx - 1]).__repr__(),
                          "2: "+''.join(
                              words[commandIdx - 1:commandIdx].__repr__())
                          )
                    input()'''
                    result = ' '.join(words[:commandIdx - 1]) + ''.join(
                        words[commandIdx - 1:commandIdx]) + ' '.join(
                        words[commandIdx:])
                    if trig in result:
                        return cls.applyWordCommands(result)
                    #words.__delitem__(commandIdx + 1)
                    #print('after',words)

                #if trig in result:
                #    return cls.applyWordCommands(' '.join(words))
                #print(words[:commandIdx-1],words[commandIdx-1:],end='.')
                #input()

                return ' '.join(words[:commandIdx - 1]) + ''.join(
                        words[commandIdx - 1:])

            else:
                return result
        else:
            return result



    @classmethod
    def applyWordCommandsOLD(cls,result: str) -> str:
        "right period period, , make period closing bracket, make period, "
        i = 0
        try:
            wordCommandIdx = result.index(WordSubs.commandTrigger,i)

        except ValueError:
            return result

        nextWords = result[wordCommandIdx:].split(' ')
        # input('next = "' + str(nextWords)+'"')
        nextWords.__delitem__(0)
        #print('frozen')
        #input(nextWords)
        #commandCount = len(nextWords)
        for w, word in enumerate(nextWords):
            # word = word.replace(',','')

            if word in WordSubs.commands.keys():
                nextWords[w] = nextWords[w].replace(word, WordSubs.commands[word], 1)
                #input('holy heck '+nextWords[w])
                commandCount = w
            else:
                break
        print('\nhere is stuff before', result, nextWords, sep=' -> ')

        result = result[:wordCommandIdx - 1]
        print('\nhere is stuff after', result, nextWords, sep=' -> ')

        try:
            result += ''.join(nextWords[:commandCount]) + ' '.join(nextWords[commandCount:])
        except UnboundLocalError:
            result += ' '.join(nextWords)
        #if commandCount == 0 or commandCount:
        #    print('\n',result,nextWords,sep=' -> ')


        if WordSubs.commandTrigger in result:
            pass#result = cls.applyWordCommands(result)
        # input(nextWords.split(' '))

        # while(WordSubs.commandTrigger in result[i:])
        return result

    @classmethod
    def isAcceptableResult(cls,result: str, debug=False) -> bool:
        return result not in WordSubs.invalidResults
        if result[-1] == ' ':
            result = result[:-1]
        if result[-1] == ',':
            result = result[:-1]
            input(result)
        truth = result not in ('', 'huh', 'butterfly')
        if debug:
            print('"' + result + '"', 'is an acceptable result ==', truth)
        return truth

    @classmethod
    def applyWordChanges(cls,result: str) -> str:

        for charIdx in range(len(result)):
            for word in WordSubs.subs.keys():
                wordLen = len(word)
                if result[charIdx: min(charIdx + wordLen, len(result))] == word:
                    result = result.replace(word, WordSubs.subs[word])
        return result

import time

# Flag to indicate the program whether should continue running.

# Our keybinding event handlers.

def display_inactive_instructions():
    message = "You are not recording. Press {} to start voice-to-keyboard".format('+'.join(bindings[0][0]))
    print('*'*len(message))
    print(message)
    print('*' * len(message))
def display_active_instructions():
    message = 'You are now recording. Voice-To-Keyboard is {} enabled.\n\tPress {} to stop the recording'.format(
        ('not' if Options.speech_to_typing_enabled else ''),'+'.join(bindings[0][0]))
    print('#' * len(message))

    print(message)

    print('You can use commands by saying:\n\t{}'.format(
        '\n\t'.join([WordSubs.commandTrigger + ' ' + word + ':\t"' + WordSubs.commands[word] + '"' for word in
                     WordSubs.commands.keys()])
    ))
    print('#' * len(message))

def display_device_info():
    device_info = sd.query_devices(1, 'input')
    print('Your system devices', device_info)
def toggle_keyboard_typing(value=None):

    #print('togglign setting')
    if value is None:
        if Options.speech_to_typing_enabled:
            Options.speech_to_typing_enabled = False
        else:
            Options.speech_to_typing_enabled = True
    else:
        Options.speech_to_typing_enabled=value
    if Options.speech_to_typing_enabled:
        B1.config(text="⏸Stop Recording")
        display_active_instructions()
        if not Options.speech_to_text_alive:
            threaded(convertMicInputToText)

        #convertMicInputToText()

        #print('You are now using voice-to-keyboard')
    else:
        B1.config(text="⏺Start Recording")
        #print('You are not using voice-to-keyboard')
        display_inactive_instructions()

def print_world():
    print("World")

def exit_application():

    stop_checking_hotkeys()
    Options.speech_to_text_alive = False

# Declare some key bindings.
# These take the format of [<key list>, <keydown handler callback>, <keyup handler callback>]
bindings = [
    [["control", "shift", "z"], None, toggle_keyboard_typing],
    [["control", "shift", "8"], None, print_world],
    [["control", "shift", "9"], None, exit_application],
]

# Register all of our keybindings
global_hotkeys.register_hotkeys(bindings)

# Finally, start listening for keypresses
global_hotkeys.start_checking_hotkeys()






q = queue.Queue()


def background(func, args):
    th = threading.Thread(target=func, args=args)
    th.start()

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

def openFile(filename):
    try:
        if sys.platform == "win32":
            os.startfile(filename)
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.call([opener, filename])
        print("opening _thisDoc:", filename)
    except:
        raise OSError("we were not able to open {}.".format(filename))



def convertMicInputToText():
    updateTextBox = Options.update_textbox_enabled
    saveToTextFile = Options.save_to_text_file
    destPath = 'testoutput.txt'
    if os.path.isfile(destPath):
        os.remove(destPath)

    #if args.samplerate is None:
    #input(sd.query_devices())
    device_info = sd.query_devices(1, 'input')
    #print('Your system devices', device_info)
    # soundfile expects an int, sounddevice provides a float:
    #samplerate = int(device_info['default_samplerate'])
    samplerate = Options.sample_rate
    global_hotkeys.restart_checking_hotkeys()
    with sd.RawInputStream(samplerate=samplerate, blocksize=8000, device=1, dtype='int16',
                           channels=1, callback=callback):


    #

        rec = KaldiRecognizer(model, samplerate)
        lineNum = 0
        while True:
            #print(lineNum)
            lineNum += 1
            if not Options.speech_to_typing_enabled:
                #print('not typing')
                break
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())['text'] #that's what I was saying .

                if WordSubs.isAcceptableResult(result):

                    result = WordSubs.applyWordCommands(WordSubs.applyWordChanges(result))
                    if result[-1].isalpha():
                        result += ' '
                    if Options.speech_to_text_enabled:
                        pyautogui.write(result)
                    print(result + '"')
                    if updateTextBox:
                        #input('"'+str(root.focus_get().())+'"')
                        result_text_box.insert(str(lineNum) + '.0', chars=result)

                    if saveToTextFile:
                        with open(destPath, 'a+') as file:
                            #lineNum += 1
                            print('in ' + destPath + ', result:', '"' + result + '" turns into "', end='')


                            file.write(result)

                            '''and then i said some other stuff, '''
                            #input(result)
                            '''for char in result:
                                pyautogui.press(char)'''

                            #result_text_box.update()
                            #result_text_box.pack()
            else:
                pass#result = rec.PartialResult()
            #print(result)
#









def convertAudioFileToText(filepath):
    
    #wf = wave.open(filepath, "rb")




    AudioSegment = pydub.AudioSegment
    sound = AudioSegment.from_file(filepath,)
    soundResampled = sound.set_frame_rate(sample_rate).set_sample_width(2).set_channels(1)
    print('starting to convert the sound.',end='')
    soundResampled.export('flattened.wav', format='wav')
    print('done.',soundResampled)
    #stream = subprocess.run(['ffmpeg','-i','test.wav','-ac 1','mono.flac'],stdout=subprocess.PIPE)
    #stream = subprocess.run(['ffmpeg'],stdout=subprocess.PIPE)                       
    '''out, _ = (ffmpeg
    .input(filepath, )
    .output('-', format='s16le', acodec='pcm_s16le', ac=1, ar='4k')
    .overwrite_output()
    .run(capture_stdout=True)
        )'''
    
    
    #input('made stream \n'+str(out))



    '''process = subprocess.Popen(['ffmpeg', '-loglevel', 'quiet', '-i',
                            filepath,
                            '-ar', str(sample_rate) , '-ac', '1', '-f', 's16le', '-'],
                            stdout=subprocess.PIPE,)'''



    #This one
    '''process = subprocess.Popen(['ffmpeg', '-loglevel', 'verbose','-i',
                                filepath, '-acodec',
                                '-ar', str(sample_rate), 'pcm_s16le', '-ac', '1', 'flattened.wav'],
                               stdout=subprocess.PIPE, )'''




    #print(filepath,'asdfasdfasdf')
    '''processArgs = ("ffmpeg -i "+filepath+" -acodec pcm_s16le -ac 1 -ar 16000 out.wav").split(' ')
    #input(processArgs)
    process = subprocess.Popen([processArgs],
                               stdout=subprocess.PIPE, )'''
    "ffmpeg -i 111.mp3 -acodec pcm_s16le -ac 1 -ar 16000 out.wav"


    #sys.stdout = open('flattened.wav', 'w')

    #sys.stdout.close()
    wf = wave.open('flattened.wav', "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            raise TypeError ("Audio file must be WAV format mono PCM.\nYou used channel {} sampwidth {} comptype {}.".format(
                wf.getnchannels(),wf.getsampwidth(),wf.getcomptype()))
            exit (1)
    result = ''
    resultLines = []
    part = 0
    while True:

        #data = process.stdout.read(4000)
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            theStr = json.loads(rec.Result())['text']
            if not WordSubs.isAcceptableResult(theStr,debug=True):
                part += 1
                continue
            #print('yo',rec.Result(),type(rec.Result()),json.loads(rec.Result()),type(json.loads(rec.Result())),sep='\n*')

            #input('part ' + str(part) + ': ' + theStr)
            #print('wtf is this json.loads("""'+rec.Result()+'""") ==', json.loads(rec.Result()))
            #input('hold it ' + str(theStr))
            resultLines.append(theStr)
            print(part,theStr)
            #result +=rec.Result()[rec.Result().index(': "') + 3:]

        else:
            pass#print(rec.PartialResult())
        part += 1
    result = ', '.join(resultLines)

    destPath = os.path.join(os.getcwd(),filepath.split('.')[0] + '.txt')

    with open(destPath,'w+') as file:
        file.write(result)
    print('saved to ',destPath)

    openFile(destPath)
    return result
    print('result ->',result)
    print('rec.FinalResult() -> ',rec.FinalResult())



display_inactive_instructions()
sample_rate=16000
# You can also specify the possible word or phrase list as JSON list, the order doesn't have to be strict
#rec = KaldiRecognizer(model, wf.getframerate(), '["oh one two three four five six seven eight nine zero", "[unk]"]')




# create the root window
root = tk.Tk()
root.title('Dre\'s Audio To Text')
root.resizable(True, True)
root.geometry('600x600')


def select_files():
    filetypes = (
        ('audio files', ('*.wav', '*.mp3','*.ogg','*.flac')),
        ('All files', '*.*')
    )

    return fd.askopenfilenames(
        title='Open multiple files',
        initialdir='/',
        filetypes=filetypes)

def doAudioToChangesBatch():
    filenames = select_files()
    for f,filename in enumerate(filenames):
        print('doing audio to changes',f,filename)

        SonicAnnotator.applyTransform(filename)
def doAudioFileToTextBatch():
    filenames = select_files()
    for f,filename in enumerate(filenames):
        print('doing audio to text',f,filename)

        convertAudioFileToText(filename)
    showinfo(
        title='Converted audio to text on these files.',
        message=filenames
    )
    

# open button
open_button = ttk.Button(
    root,
    text='Open Files Verbially',
    #command=select_files
    command=lambda : threaded(doAudioFileToTextBatch)
)

open_button.pack(expand=True)

analyse_notes_open_button = ttk.Button(
    root,
    text='Open Files Musically',
    #command=select_files
    command=lambda : threaded(doAudioToChangesBatch)
)
analyse_notes_open_button.pack()
def hello():

   tk.messagebox.showinfo("Yer done bud", "Hello World")
result_text_box = tk.Text(root, height = 30, width = 80)
result_text_box.pack(expand=True)
result_text_box.insert('1.0', '')
scrollbar = tk.Scrollbar(root,command=result_text_box.yview)
scrollbar.pack( side = tk.RIGHT, fill = tk.Y )
result_text_box['yscrollcommand'] = scrollbar.set
result_text_box.grid_columnconfigure(0, weight=1)
#B1 = tk.Button(root, text = "Say Hello", command = convertMicInputToText)
#B1 = tk.Button(root, text = "Say Hello", command = lambda : background(convertMicInputToText, tuple()))
def toggleTypingVariable():
    Options.speech_to_typing_enabled = not Options.speech_to_typing_enabled
#B1 = tk.Button(root, text = "Toggle Recording", command = root.after(0,toggleTypingVariable))
B1 = tk.Button(root, text = "Start Recording", command = lambda : threaded(toggle_keyboard_typing))

#b1 = tk.Button(root,text='copy',command = lambda : background(print_numbers, (50,))) #args must be a tuple even if it is only one
B1.pack()
usingKeyboardCheckbox = IntVar()
def set_keyboard_enabled():

    Options.speech_to_text_enabled = usingKeyboardCheckbox.get()
    print('Voice to keyboard has been {}abled'.format(
        'en' if Options.speech_to_text_enabled else 'dis'
    ))
    #toggle_keyboard_typing(Options.speech_to_text_enabled)


c1 = tk.Checkbutton(root, text='Text To Keyboard',variable=usingKeyboardCheckbox, onvalue=1, offvalue=0, command=lambda:threaded(set_keyboard_enabled))
if Options.speech_to_typing_enabled:
    c1.select()
c1.pack()

rec = KaldiRecognizer(model, sample_rate,)
root.mainloop()

















filepath = "test.wav"
if not os.path.exists("model"):
    print ("Please download the model from https://alphacephei.com/vosk/models and unpack as 'model' in the current folder.")
    exit (1)

