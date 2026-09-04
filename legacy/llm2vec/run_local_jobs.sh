for script in ./grid_search/jobs_local/*.sh; do
    echo "Running $script"
    bash "$script"
done

