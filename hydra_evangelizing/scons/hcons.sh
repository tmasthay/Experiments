hcons () 
{ 
    local HELP=0;
    local CLEAN=0;
    for arg in "$@";
    do
        if [ "$arg" == "--help" ]; then
            HELP=1;
        else
            if [ "$arg" == "-c" ]; then
                CLEAN=1;
            fi;
        fi;
    done;
    if [ $CLEAN -eq 1 ]; then
        scons -c;
    else
        python3 hcons.py "$@";
        if [ $HELP -eq 0 && $? -eq 0 ]; then
            scons;
        elif [ $HELP -eq 0 ]; then
            echo "Invalid hydra config...skipping scons";
            return 1
        fi
    fi
}