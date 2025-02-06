# Graph
Sick of slow, laggy charts? Graph is the answer.  
Created on python 3.11, but likely to run on older versions.  
  
# Description  
Primitive chart software for python.  
Charts sequential lines, primarily. Other primitives like arbitrary lines and dots work, but are far from fully performance optimized. A sequential line is a line where  
    x_current >= x_previous  
The main goals are:  
    + speed, primarily when running, but launch speed is in the works  
    + ability to easily identify underlying data when necessary.  
    + avoiding the use qt and similar outdated technologies.  
    + simple code  
  
# TODO  
[ ] Reduce launchtime  
[ ] Implement non-sequential line resampling  
[ ] Tests  
[ ] For some reason, resampling in bg thread blocks it from running, making  
launchtime really slow  
[ ] Sometimes, there is a tiny bit of lag when a new resample comes along  
[ ] Advanced cutting strategies  
[ ] Code cleanup  
[ ] cuts should include an extra datapoint to avoid lines disappearing when a  
point is out of range  
[ ] better, non-cartoony zoom and movement  
[ ] the first movement after opening a chart has a weird small pause before it occurs  
[ ] something about the cached resamples in another thread doesn't work fully,  
as sometimes loading a cached resample takes weirdly long  
[ ] memory optimization - RAM starts filling up at about 100 million values,  
doesn't seem like it should as 10M is 700 megs  
  
  
# Notes  
Built on pygame + numpy + numba  
  
# Possible roadmaps forward  
[ ] Implement image charting  
[ ] Centralize cutting + resampling decisions in Graph outside of GraphObjects  
    -> This has some cons, have to think about it  
    -> This has the pro of enabling further optimizations  
[ ] After any cut/resample cycle, convert the entire chart to an image and only  
translate/zoom said image until a new cut/resample cycle.  
[ ] Perhaps data manipulations can be centralized as well? Or done by GPU given  
their nature?  
