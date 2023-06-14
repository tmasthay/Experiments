#see fonts
see_fonts(){
    fc-list | cut -f2 -d':' | sort -u
}

#conda environment helpers
prepend_env_var() {
    # Check if the environment variable is already set
    if [ -z "${!1}" ]; then
        # If not, set the environment variable to the new value without a trailing colon
        export $1=$2
    else
        # If yes, store the old value and prepend the new value with a colon separator
        export OLD_$1=${!1}
        export $1=$2:${!1}
    fi
}

revert_env_var() {
    local old_var="OLD_$1"
    eval "export $1=\"\${$old_var}\""

    # Unset the old value
    unset OLD_$1
}

