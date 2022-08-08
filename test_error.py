import os
import sys
import time
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

import paddle
from paddle.inference import create_predictor, Config

import predictor
import clicker 

#GPU下进行预测

def infer_image(image_path=None, click_position=None, positive_click=True, norm_radius=2, spatial_scale=1.0, pred_thr=0.49, device=None):
    """
    """
    image_path = 'Case59.nii.gz'
    model_path, param_path = "output_cpu/static_Vnet_model.pdmodel",  \
                             "output_cpu/static_Vnet_model.pdiparams"
    new_shape = (512, 512, 12) # xyz #这个形状与训练的对数据预处理的形状要一致
    paddle.device.set_device(device)

    if click_position is None:
        click_position = [[234, 284, 7, 100]]

    def resampleImage(refer_image, out_size, out_spacing=None, interpolator=sitk.sitkLinear):
        #根据输出图像，对SimpleITK 的数据进行重新采样。重新设置spacing和shape
        if out_spacing is None:
            out_spacing = tuple((refer_image.GetSize() / np.array(out_size)) * refer_image.GetSpacing()) 
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(refer_image)  
        resampler.SetSize(out_size)
        resampler.SetOutputSpacing(out_spacing)
        resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
        resampler.SetInterpolator(interpolator)
        return resampler.Execute(refer_image), out_spacing

    def crop_wwwc(sitkimg, max_v,min_v):
        #对SimpleITK的数据进行窗宽窗位的裁剪，应与训练前对数据预处理时一致
        intensityWindow = sitk.IntensityWindowingImageFilter()
        intensityWindow.SetWindowMaximum(max_v)
        intensityWindow.SetWindowMinimum(min_v)
        return intensityWindow.Execute(sitkimg)

    def GetLargestConnectedCompont(binarysitk_image):
        # 最大连通域提取,binarysitk_image 是掩膜
        cc = sitk.ConnectedComponent(binarysitk_image)
        stats = sitk.LabelIntensityStatisticsImageFilter()
        stats.SetGlobalDefaultNumberOfThreads(8)
        stats.Execute(cc, binarysitk_image)#根据掩膜计算统计量
        # stats.
        maxlabel = 0
        maxsize = 0
        for l in stats.GetLabels():#掩膜中存在的标签类别
            size = stats.GetPhysicalSize(l)
            if maxsize < size:#只保留最大的标签类别
                maxlabel = l
                maxsize = size
        labelmaskimage = sitk.GetArrayFromImage(cc)
        outmask = labelmaskimage.copy()
        if len(stats.GetLabels()):
            outmask[labelmaskimage == maxlabel] = 255
            outmask[labelmaskimage != maxlabel] = 0
        return outmask
        
    # 预处理
    origin = sitk.ReadImage(image_path)
    itk_img_res = crop_wwwc(origin, max_v=2650, min_v=0) # 和预处理文件一致 (512, 512, 12) WHD
    itk_img_res, new_spacing = resampleImage(itk_img_res, out_size=new_shape)  # 得到重新采样后的图像 origin: (880, 880, 12)
    npy_img = sitk.GetArrayFromImage(itk_img_res).astype("float32") # 12, 512, 512 DHW

    input_data = np.expand_dims(np.transpose(npy_img, [2, 1, 0]), axis=0)
    if input_data.max() > 0: # 归一化
        input_data = input_data / input_data.max()

    print(f"输入网络前数据的形状:{input_data.shape}") # shape (1, 512, 512, 12)
    
    ##################### 预测部分看这里 #########################3
    
    #创建预测器，加载模型进行预测，初始化一次
    predictor_params_ = {'norm_radius': norm_radius, "spatial_scale": spatial_scale} # 默认和训练一样
    inference_predictor = predictor.BasePredictor(model_path, param_path, with_flip=False, device=device, **predictor_params_)# 
    # 根据输入初始化
    inference_predictor.set_input_image(input_data)

    for i, c in enumerate(click_position): # input a number 
        _, _, _, tag = c
        click = clicker.Click(is_positive=tag>0, coords=c)
        pred_probs = inference_predictor.get_prediction_noclicker(click)
        output_data = (pred_probs > pred_thr) * pred_probs #  (12, 512, 512) DHW
        output_data[output_data>0] = 1
    
    ##################### 预测部分看这里 #########################3
    
    

    # 加载3d模型预测的mask，由numpy 转换成SimpleITK格式
    output_data = np.transpose(output_data, [2, 1, 0])  
    mask_itk_new = sitk.GetImageFromArray(output_data) # (512, 512, 12) WHD
    mask_itk_new.SetSpacing(new_spacing)
    mask_itk_new.SetOrigin(origin.GetOrigin())
    mask_itk_new.SetDirection(origin.GetDirection())
    mask_itk_new = sitk.Cast(mask_itk_new, sitk.sitkUInt8)
    
    # 暂时没有杂散目标，不需要最大联通域提取
    Mask, _ = resampleImage(mask_itk_new, origin.GetSize(), origin.GetSpacing(), sitk.sitkNearestNeighbor)
    Mask.CopyInformation(origin)
    sitk.WriteImage(Mask, image_path.replace('.nii.gz','_predict_np_sitk.nii.gz'))
    print("预测成功！save to {}".format(image_path.replace('.nii.gz','_predict_np.nii.gz')))

if __name__ == "__main__":
    device = "cpu"
    infer_image(device=device)
