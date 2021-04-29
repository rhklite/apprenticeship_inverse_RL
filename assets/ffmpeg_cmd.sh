# source: https://stackoverflow.com/a/60569422/10251025

ffmpeg -i Expert.mp4 -stream_loop -1 -i Student_0.mp4 -stream_loop -1 -i Student_2.mp4  -i  Student_9.mp4 -filter_complex \
"[0]drawtext=text='Expert':fontsize=20:x=(w-text_w)/2-200:y=(h-text_h)/2-150[v0];
 [1]drawtext=text='Student 0':fontsize=20:x=(w-text_w)/2-200:y=(h-text_h)/2-150[v1];
 [2]drawtext=text='Student 2':fontsize=20:x=(w-text_w)/2-200:y=(h-text_h)/2-150[v2];
 [3]drawtext=text='Student 9':fontsize=20:x=(w-text_w)/2-200:y=(h-text_h)/2-150[v3];
 [v0][v1][v2][v3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0:shortest=1" -y output.mp4

ffmpeg -i output.mp4 -vf "fps=50" -loop 0 -y output.gif

# rm output.mp4