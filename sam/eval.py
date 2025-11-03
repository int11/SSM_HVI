import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data import *
from loss.losses import *
from net.CIDNet import CIDNet
from measure import metrics
import dist
from data.options import option, load_datasets
from net.CIDNet_SSM import CIDNet as CIDNet_SSM
import safetensors.torch as sf
from huggingface_hub import hf_hub_download
import torchvision.transforms as transforms


def eval(model, testing_data_loader, alpha_predict=True, base_alpha_s=1.0, base_alpha_i=1.0):
    torch.set_grad_enabled(False)
    
    model = dist.de_parallel(model)
    model.eval()

    output_list = []  # 출력 이미지 저장용 리스트
    gt_list = []   # 라벨 이미지 저장용 리스트

    for batch in testing_data_loader:
        with torch.no_grad():
            input, gt, name = batch[0], batch[1], batch[2]
            input = input.cuda()
            output = model(input, alpha_predict=alpha_predict, base_alpha_s=base_alpha_s, base_alpha_i=base_alpha_i)
        output = torch.clamp(output.cuda(),0,1).cuda()
        output_np = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        output_list.append(output_np)
        # gt는 tensor이므로 PIL로 변환
        from torchvision.transforms import ToPILImage
        gt_img = ToPILImage()(gt.squeeze(0).cpu())
        gt_list.append(gt_img)
        torch.cuda.empty_cache()
    
    torch.set_grad_enabled(True)

    return output_list, gt_list


def load_cidnet_base_model(model_path, device):
    """Hugging Face에서 CIDNet 모델을 다운로드하고 로드"""
    print(f"Loading CIDNet model from: {model_path}")
    
    # Hugging Face Hub에서 CIDNet model 다운로드
    model_file = hf_hub_download(
        repo_id=model_path, 
        filename="model.safetensors", 
        repo_type="model"
    )
    print(f"CIDNet model downloaded from: {model_file}")
    
    # 모델 초기화 및 가중치 로드
    model = CIDNet()
    state_dict = sf.load_file(model_file)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def eval_original(model, testing_data_loader, device, base_alpha_s=1.0, base_alpha_i=1.0):
    """기본 CIDNet으로 dataloader의 모든 이미지 처리"""
    torch.set_grad_enabled(False)
    
    model = dist.de_parallel(model)
    model.eval()
    
    # 기본 CIDNet의 alpha 파라미터 설정
    model.trans.alpha_s = base_alpha_s
    model.trans.alpha_i = base_alpha_i

    output_list = []  # 출력 이미지 저장용 리스트

    for batch in testing_data_loader:
        with torch.no_grad():
            input, gt, name = batch[0], batch[1], batch[2]
            input = input.to(device)
            output = model(input)
        output = torch.clamp(output, 0, 1).to(device)
        output_np = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        output_list.append(output_np)
        torch.cuda.empty_cache()
    
    torch.set_grad_enabled(True)

    return output_list
    
if __name__ == '__main__':
    parser = option()
    parser.add_argument('--weight_path', type=str, default='weights/train2025-10-13-005336/epoch_1500.pth', help='Path to the pre-trained model weights')
    parser.add_argument('--output_dir', type=str, default='results/ssm_eval_results', help='Directory to save comparison images')
    parser.add_argument('--cidnet_model', type=str, default="Fediory/HVI-CIDNet-LOLv1-woperc",
                        help='CIDNet model name or path from Hugging Face')
    parser.add_argument('--base_alpha_s', type=float, default=1.0, help='Base alpha_s parameter for CIDNet')
    parser.add_argument('--base_alpha_i', type=float, default=1.0, help='Base alpha_i parameter for CIDNet')

    args = parser.parse_args()

    training_data_loader, testing_data_loader = load_datasets(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load CIDNet_sam model
    eval_net = CIDNet_SSM().cuda()
    checkpoint_data = torch.load(args.weight_path, map_location=lambda storage, loc: storage)
    eval_net.load_state_dict(checkpoint_data['model_state_dict'])
    print(f"Loaded CIDNet_SSM checkpoint from {args.weight_path}")

    # Load base CIDNet model
    cidnet_base = load_cidnet_base_model(args.cidnet_model, device)
    print(f"Loaded base CIDNet model from {args.cidnet_model}")

    # Evaluate - CIDNet_sam with alpha prediction
    output_list, gt_list = eval(eval_net, testing_data_loader, alpha_predict=True, base_alpha_s=args.base_alpha_s, base_alpha_i=args.base_alpha_i)
    
    # Evaluate - Base CIDNet (without alpha prediction)
    output_base_list = eval_original(cidnet_base, testing_data_loader, device, base_alpha_s=args.base_alpha_s, base_alpha_i=args.base_alpha_i)
    
    # Calculate metrics for CIDNet_sam
    print("\n" + "="*60)
    print("CIDNet_SSM with Alpha Prediction")
    print("="*60)
    avg_psnr_sam, avg_ssim_sam, avg_lpips_sam = metrics(output_list, gt_list, use_GT_mean=args.use_GT_mean)
    print(f"PSNR: {avg_psnr_sam:.4f} dB || SSIM: {avg_ssim_sam:.4f} || LPIPS: {avg_lpips_sam:.4f}")
    
    # Calculate metrics for Base CIDNet
    print("\n" + "="*60)
    print("Base CIDNet (Standard Parameters)")
    print("="*60)
    avg_psnr_base, avg_ssim_base, avg_lpips_base = metrics(output_base_list, gt_list, use_GT_mean=args.use_GT_mean)
    print(f"PSNR: {avg_psnr_base:.4f} dB || SSIM: {avg_ssim_base:.4f} || LPIPS: {avg_lpips_base:.4f}")
    
    # Print comparison
    print("\n" + "="*60)
    print("Performance Comparison")
    print("="*60)
    print(f"PSNR Improvement: {avg_psnr_sam - avg_psnr_base:+.4f} dB")
    print(f"SSIM Improvement: {avg_ssim_sam - avg_ssim_base:+.4f}")
    print(f"LPIPS Improvement: {avg_lpips_sam - avg_lpips_base:+.4f}")
    
    # Save comparison images
    os.makedirs(args.output_dir, exist_ok=True)
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    
    # Get input images from dataloader
    input_images = []
    for batch in testing_data_loader:
        input_tensor = batch[0]
        input_np = input_tensor.squeeze(0).numpy().transpose(1, 2, 0)
        input_images.append(input_np)
    
    for idx, (output_np, base_output_np, gt_img, input_np) in enumerate(zip(output_list, output_base_list, gt_list, input_images)):
        # Convert numpy outputs to PIL
        output_img = Image.fromarray((output_np * 255).astype(np.uint8))
        base_output_img = Image.fromarray((base_output_np * 255).astype(np.uint8))
        input_img = Image.fromarray((input_np * 255).astype(np.uint8))
        
        # Create comparison image (Input | Base CIDNet | CIDNet_sam | GT)
        h, w = output_np.shape[:2]
        comparison = Image.new('RGB', (w * 4, h + 40))
        comparison.paste(input_img, (0, 40))
        comparison.paste(base_output_img, (w, 40))
        comparison.paste(output_img, (w * 2, 40))
        comparison.paste(gt_img, (w * 3, 40))
        
        # Add labels
        draw = ImageDraw.Draw(comparison)
        font = ImageFont.load_default()
        label_y = 10
        draw.text((w//2 - 30, label_y), "Input", fill="white", font=font)
        draw.text((w + w//2 - 50, label_y), "Base CIDNet", fill="white", font=font)
        draw.text((w*2 + w//2 - 40, label_y), "CIDNet_SSM", fill="white", font=font)
        draw.text((w*3 + w//2 - 20, label_y), "GT", fill="white", font=font)
        
        # Save comparison image
        comparison_path = os.path.join(args.output_dir, f'comparison_{idx+1:03d}.png')
        comparison.save(comparison_path)
        print(f"Saved comparison image: {comparison_path}")
    
    print(f"\n✓ Saved {len(output_list)} comparison images to: {args.output_dir}")