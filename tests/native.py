"""Native ABI integration test. Run with CPython; OpenCV Python bindings are not needed."""
import ctypes,json,os
from pathlib import Path
root=Path(__file__).resolve().parent.parent
directory=root/'runtimes/windows'
with os.add_dll_directory(str(directory)):
 lib=ctypes.CDLL(str(directory/'aqua_opencv.dll'))
lib.cv_command.argtypes=[ctypes.c_void_p,ctypes.c_int32,ctypes.c_void_p,ctypes.c_int32]
lib.cv_decode.argtypes=lib.cv_command.argtypes;lib.cv_upload.argtypes=lib.cv_command.argtypes;lib.cv_png.argtypes=lib.cv_command.argtypes
def call(name,data):
 output=ctypes.create_string_buffer(1048576);result=getattr(lib,name)(data,len(data),output,len(output))
 assert result>=0,(name,result)
 return output.raw[:result]
def command(q):
 response=json.loads(call('cv_command',json.dumps(q).encode()))
 assert response['ok'],(q,response)
 return response['value']
sid=command({'op':'open'})
def request(**q):return command(dict(q,session=sid))
try:
 info=request(op='info');print(info['version'],info['backend'])
 sample=request(op='sample',frame=12)
 for kind,name in [(1,'face.xml'),(2,'yunet.onnx')]:call('cv_upload',sid.to_bytes(4,'little')+bytes([kind])+(root/'assets'/name).read_bytes())
 portrait=int.from_bytes(call('cv_decode',sid.to_bytes(4,'little')+(root/'assets/portrait.jpg').read_bytes()),'little')
 failures=[]
 for feature in info['features']:
  try:
   result=request(op='apply',image=portrait if feature in ['face','dnn'] else sample,feature=feature,parameters={})
   assert request(op='inspect',image=result['image'])['width']>0
   if feature in ['face','dnn']:assert result['stats']['faces']>=1
   if feature=='qr':assert result['stats']['text']=='https://opencv.org'
   if feature=='calibration':assert abs(result['stats']['z']-15)<.001
   if feature=='template':assert result['stats']['score']>.99
   print('PASS',feature,round(result['milliseconds'],2),result['stats'])
   request(op='release',image=result['image'])
  except Exception as e:failures.append(str(e))
 request(op='release',image=sample);request(op='release',image=portrait)
 assert request(op='info')['images']==0
 assert not failures,'\n'.join(failures)
 print('OPENCV_NATIVE_OK',len(info['features']))
finally:request(op='close')
