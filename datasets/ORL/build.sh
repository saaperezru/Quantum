for i in {1..9}
do
    cd s$i
    for j in {1..9}
    do
        mv $j.pgm ../0$i.0$j.pgm
    done
    mv 10.pgm ../0$i.10.pgm
    cd ..
    rmdir s$i
done
for i in {10..40}
do
    cd s$i
    for j in {1..9}
    do
        mv $j.pgm ../$i.0$j.pgm
    done
    mv 10.pgm ../$i.10.pgm
    cd ..
    rmdir s$i
done

