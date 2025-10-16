from curses.ascii import islower
import torch
import torchvision

# Text formatting.
bold = "\033[1m"
end = "\033[0m"

allModules = dir(torchvision.models)
skip = 0
common = ["detection", "get_model"]
# Navigate the torchvision.models namespace and display the available models.
for en, module in enumerate(allModules):
    if module[0].islower():
        if skip == 0:
            # Handle special case for vision_transformer.
            if module == "vision_transformer":
                print(f"Model Family: {bold}{module}{end}")
                for mod in allModules[en + 1 :]:
                    if "vit" in mod:
                        print(f"\t  |__ {mod}")
                        skip += 1

            # Handle special case for inception.
            elif module == "inception":
                print(f"Model Family: {bold}{module}{end}")
                for mod in allModules[en + 1 :]:
                    if module in mod:
                        print(f"\t  |__ {mod}")
                        skip += 1

            # Handle special case for shufflenet.
            elif "shufflenet" in module:
                module = "shufflenet"
                print(f"Model Family: {bold}{module}{end}")
                for mod in allModules[en + 1 :]:
                    if "shufflenet" in mod:
                        print(f"\t  |__ {mod}")
                        skip += 1

            # Handle special case for swin.
            elif "swin" in module:
                module = "swin_transformer"
                print(f"Model Family: {bold}{module}{end}")
                for mod in allModules[en + 1 :]:
                    if ("swin" in mod) and (mod != "swin_transformer"):
                        print(f"\t  |__ {mod}")
                        skip += 1
                    elif mod == "swin_transformer":
                        skip += 1

            # General cases.
            else:
                if module not in common:
                    count = 0
                    for mod in allModules[en + 1 :]:
                        if module in mod:
                            count += 1
                            if count > 1:
                                break
                    if count > 1:
                        print(f"Model Family: {bold}{module}{end}")
                        for mod in allModules[en + 1 :]:
                            if module in mod:
                                print(f"\t  |__ {mod}")
                                skip += 1

                else:
                    for mod in allModules[en + 1 :]:
                        if module in mod:
                            skip += 1

        else:
            skip -= 1
