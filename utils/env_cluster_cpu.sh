env_dir=./env_cpu
python_version="3.6.1"

module load eth_proxy python_cpu/$python_version hdf5/1.10.1

if [ ! -d "$env_dir" ]; then
    python -m pip install --user virtualenv
    python -m virtualenv \
        --system-site-packages \
        --python="/cluster/apps/python/$python_version/bin/python" \
        "$env_dir"
fi

source "$env_dir/bin/activate"
