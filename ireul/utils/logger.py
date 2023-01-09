import os
import uuid

from ireul.utils.io import create_dir


def log_path():
    import ireul

    log_path = os.path.abspath(
        os.path.join(ireul.__file__, "../../", "offlinerl_tmp")
    )

    create_dir(log_path)

    return log_path
