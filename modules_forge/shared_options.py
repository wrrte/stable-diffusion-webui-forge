def register(options_templates, options_section, OptionInfo):
    options_templates.update(options_section((None, "Forge Hidden options"), {
        "forge_unet_storage_dtype": OptionInfo('Automatic'),
        "forge_inference_memory": OptionInfo(4096),
        "forge_async_loading": OptionInfo('Queue'),
        "forge_pin_shared_memory": OptionInfo('CPU'),
        "forge_preset": OptionInfo('sd'),
        "forge_additional_modules": OptionInfo(['/Users/choemj/Painting/stable-diffusion-webui-forge/models/VAE/fixFP16ErrorsSDXLLowerMemoryUse_v10.safetensors']),
    }))
    options_templates.update(options_section(('ui_alternatives', "UI alternatives", "ui"), {
        "forge_canvas_plain": OptionInfo(False, "ForgeCanvas: use plain background").needs_reload_ui(),
        "forge_canvas_toolbar_always": OptionInfo(False, "ForgeCanvas: toolbar always visible").needs_reload_ui(),
    }))
